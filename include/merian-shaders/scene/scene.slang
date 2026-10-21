__exported import merian_shaders.scene.camera;
__exported import merian_shaders.scene.acceleration_structure;
__exported import merian_shaders.scene.environment_map;
__exported import merian_shaders.shading.materials.material_system;
__exported import merian_shaders.scene.scene_data;
__exported import merian_shaders.scene.hit;
__exported import merian_shaders.scene.nee;

import merian_shaders.colors.colorspaces;
import merian_shaders.colors.debug;
import merian_shaders.utils.encoding;
import merian_shaders.utils.hash;
import merian_shaders.utils.pseudorandom;
import merian_shaders.utils.random;
import merian_shaders.utils.sampling;

// Workaround for https://github.com/shader-slang/slang/pull/10769; scene.cpp injects replacements.
__exported import scene_as_workaround;
__exported import scene_env_map_workaround;
__exported import scene_prev_env_map_workaround;
__exported import scene_volume_workaround;

namespace merian {

// Once https://github.com/shader-slang/slang/pull/10769 lands these can replace the workaround
// imports above. extern struct SceneAccelerationStructure : AccelerationStructure =
// NullAccelerationStructure; extern struct SceneEnvMap : IEnvMap = EmptyEnv; extern struct
// ScenePrevEnvMap : IEnvMap = EmptyEnv;

// set by Scene from the active camera
extern static const bool merian_hint_enable_thin_lens = false;

struct Scene {
    SceneAccelerationStructure as;
    SceneAccelerationStructure prev_as;

    StructuredBuffer<GeometryData, ScalarDataLayout> geometries;

    // One transform per TLAS instance, indexed by InstanceIndex().
    StructuredBuffer<float4x4, ScalarDataLayout> instance_transforms;
    StructuredBuffer<float4x4, ScalarDataLayout> inverse_transposed_instance_transforms;
    StructuredBuffer<float4x4, ScalarDataLayout> prev_instance_transforms;
    StructuredBuffer<float4x4, ScalarDataLayout> prev_inverse_transposed_instance_transforms;

    StructuredBuffer<GeometryRemap, ScalarDataLayout> prev_geometry_remap;
    uint prev_geometry_remap_count;

    ParameterBlock<MaterialSystem> material_system;
    SceneEnvMap env_map;
    ScenePrevEnvMap prev_env_map;
    NEE nee;
    // The previous frame's, looked up by this frame's GeometryIDs.
    NEE prev_nee;

    Camera camera;
    Camera prev_camera;

    uint frame;
    uint seed;
    float time;
    float time_diff;
    float prev_time;

    // the medium outside any surface; a vacuum type unless the scene sets one
    SceneHomogeneousVolume exterior_volume;

    PathMedium<SceneHomogeneousVolume> make_path_medium() {
        return PathMedium<SceneHomogeneousVolume>(exterior_volume);
    }

    Hit trace_ray(const RayDesc ray,
                  const uint ray_flags = RAY_FLAG_NONE,
                  const uint instances_mask = 0xff,
                  const bool prev_frame = false) {
        SceneAccelerationStructure::RayQueryImpl rq;
        if (prev_frame)
            prev_as.trace_ray(rq, ray, ray_flags, instances_mask);
        else
            as.trace_ray(rq, ray, ray_flags, instances_mask);
        _proceed_ray(rq, prev_frame);

        if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            const Hit hit = Hit(rq.CommittedInstanceID(), rq.CommittedInstanceIndex(),
                                rq.CommittedPrimitiveIndex(), rq.CommittedGeometryIndex(),
                                rq.CommittedRayBarycentrics());
            return prev_frame ? from_prev_frame(hit) : hit;
        }

        return Hit();
    }

    // true if nothing was hit
    bool trace_visibility_ray(const RayDesc ray,
                              const uint ray_flags = RAY_FLAG_NONE,
                              const uint instances_mask = 0xff) {
        SceneAccelerationStructure::RayQueryImpl rq;
        as.trace_ray(rq, ray, ray_flags, instances_mask);
        _proceed_ray(rq);

        return rq.CommittedStatus() == COMMITTED_NOTHING;
    }

    // Primary camera ray for a pixel. The lens sample depends on (pixel, frame, seed) only, so
    // every pass that calls this within a frame gets the identical ray.
    RayDesc generate_camera_ray(const uint2 pixel, const uint2 frame_dimensions) {
        if (merian_hint_enable_thin_lens && camera.aperture_radius > 0.f) {
            // R2 low-discrepancy sequence over frames, per-pixel rotated to decorrelate pixels.
            RandomGenerator rng = PCG(xxhash32(uint3(pixel, seed)));
            const float2 random = frac(r2_sequence(frame) + rng.next_float2());
            return camera.generate_ray_thinlens(pixel, frame_dimensions, random);
        }
        return camera.generate_ray_pinhole(pixel, frame_dimensions);
    }

    GeometryData get_geometry_data(const GeometryID geometry_id) {
        return geometries[geometry_id];
    }

    // the hit must be at most one frame old
    Hit from_prev_frame(const Hit hit) {
        if (prev_geometry_remap_count == 0 || !hit.is_valid())
            return hit;
        const GeometryRemap remap = prev_geometry_remap[hit.get_geometry_id()];
        Hit mapped = hit;
        mapped.instance_id = remap.geometry_id;
        mapped.geometry_index = 0;
        mapped.instance_index = remap.instance_index;
        return mapped;
    }

    // A miss or a sky portal: the hit sees the environment.
    bool is_distant(const Hit hit) {
        return !hit.is_valid() ||
               (get_geometry_data(hit.get_geometry_id()).flags & GeometryDataFlags::UseEnvMap) != 0;
    }

    // misses exist in both frames
    bool exists(const Hit hit, const bool prev_frame) {
        if (!hit.is_valid())
            return true;
        const GeometryDataFlags absent =
            prev_frame ? GeometryDataFlags::New : GeometryDataFlags::Retired;
        return (get_geometry_data(hit.get_geometry_id()).flags & absent) == 0;
    }

    float get_time(const bool prev_frame) {
        return prev_frame ? prev_time : time;
    }

    GeometryData get_geometry_data(const uint instance_id, const uint geometry_index) {
        return get_geometry_data(instance_id + geometry_index);
    }

    float4x4 get_instance_transform(const uint instance_index, const bool prev_frame = false) {
        return prev_frame ? prev_instance_transforms[instance_index]
                          : instance_transforms[instance_index];
    }

    float4x4 get_inverse_transposed_instance_transform(const uint instance_index,
                                                       const bool prev_frame = false) {
        return prev_frame ? prev_inverse_transposed_instance_transforms[instance_index]
                          : inverse_transposed_instance_transforms[instance_index];
    }

    MaterialID get_material_id(const GeometryID geometry_id) {
        return get_geometry_data(geometry_id).material_id;
    }

    MaterialID get_material_id(const Hit hit) {
        return get_material_id(hit.get_geometry_id());
    }

    uint3 get_vertex_indices(const GeometryData geometry_data, const uint primitive_id) {
        return merian::get_vertex_indices(geometry_data, primitive_id);
    }

    PackedVertexData get_packed_vertex_data(const GeometryData geometry_data,
                                            const uint vertex_index) {
        return merian::get_packed_vertex_data(geometry_data, vertex_index);
    }

    VertexBufferAddress get_vertices(const GeometryData geometry_data, const bool prev_frame) {
        return prev_frame ? geometry_data.prev_vertices : geometry_data.vertices;
    }

    float2 get_interpolated_uv(const GeometryData geometry_data,
                               const uint primitive_id,
                               const float3 barycentrics,
                               const bool prev_frame = false) {
        const uint3 indices = get_vertex_indices(geometry_data, primitive_id);
        const VertexBufferAddress vertices = get_vertices(geometry_data, prev_frame);

        return barycentrics.x * float2(vertices[indices.x].uv) +
               barycentrics.y * float2(vertices[indices.y].uv) +
               barycentrics.z * float2(vertices[indices.z].uv);
    }

    float3 to_world_position(const GeometryData gd,
                             const uint instance_index,
                             const float3 object_pos,
                             const bool prev_frame = false) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_pos;
        return mul(get_instance_transform(instance_index, prev_frame), float4(object_pos, 1.0)).xyz;
    }

    float3 to_world_normal(const GeometryData gd,
                           const uint instance_index,
                           const float3 object_normal,
                           const bool prev_frame = false) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_normal;
        return normalize(mul(get_inverse_transposed_instance_transform(instance_index, prev_frame),
                             float4(object_normal, 0.0))
                             .xyz);
    }

    float3 to_world_vector(const GeometryData gd,
                           const uint instance_index,
                           const float3 object_vec,
                           const bool prev_frame = false) {
        if (gd.flags & GeometryDataFlags::Pretransformed)
            return object_vec;
        return mul((float3x3)get_instance_transform(instance_index, prev_frame), object_vec);
    }

    Optional<GeometryData> get_geometry_data(const Hit hit) {
        if (!hit.is_valid())
            return none;
        return get_geometry_data(hit.instance_id, hit.geometry_index);
    }

    // Hit must be a valid triangle hit.
    TriangleHit get_triangle_hit(const Hit hit, const bool prev_frame = false) {
        return get_triangle_hit(hit, get_geometry_data(hit.instance_id, hit.geometry_index),
                                prev_frame);
    }

    TriangleHit
    get_triangle_hit(const Hit hit, const GeometryData gd, const bool prev_frame = false) {
        const uint3 indices = get_vertex_indices(gd, hit.primitive_id);
        const VertexBufferAddress vertices = get_vertices(gd, prev_frame);
        TriangleHit tri;
        tri.gd = gd;
        tri.v0 = vertices[indices.x];
        tri.v1 = vertices[indices.y];
        tri.v2 = vertices[indices.z];
        return tri;
    }

    // tri of the other frame, reused where its vertices did not change
    TriangleHit get_triangle_hit(const Hit hit, const TriangleHit tri, const bool prev_frame) {
        if (has_static_vertices(tri.gd))
            return tri;
        return get_triangle_hit(hit, tri.gd, prev_frame);
    }

    static bool has_static_vertices(const GeometryData gd) {
        return (uint64_t)gd.prev_vertices == (uint64_t)gd.vertices;
    }

    // normal curvatures across and along wi's plane of incidence, positive bulging towards wi
    float2 get_curvatures(const Hit hit, const TriangleHit tri, const float3 wi) {
        if (tri.gd.flags & GeometryDataFlags::FlatShading) {
            return float2(0.f);
        }
        const uint instance = hit.instance_index;
        const float3 n0 =
            to_world_normal(tri.gd, instance, decode_normal(tri.v0.encoded_normal));
        const float3 dn1 =
            to_world_normal(tri.gd, instance, decode_normal(tri.v1.encoded_normal)) - n0;
        const float3 dn2 =
            to_world_normal(tri.gd, instance, decode_normal(tri.v2.encoded_normal)) - n0;
        const float3 e1 = to_world_vector(tri.gd, instance, tri.get_triangle_edge01());
        const float3 e2 = to_world_vector(tri.gd, instance, tri.get_triangle_edge02());
        const float3 w = normalize(cross(e1, e2));
        const float3 u = normalize(e1);
        const float3 v = cross(w, u);

        // S E = D for the edges E and normal turns D in the face's plane
        const float det = dot(e1, u) * dot(e2, v) - dot(e2, u) * dot(e1, v);
        if (!(abs(det) > 0.f)) {
            return float2(0.f);
        }
        const float2x2 e_inv = float2x2(dot(e2, v), -dot(e2, u), -dot(e1, v), dot(e1, u)) / det;
        const float2x2 d = float2x2(dot(dn1, u), dot(dn2, u), dot(dn1, v), dot(dn2, v));
        const float2x2 shape = mul(d, e_inv);
        const float2x2 symmetric = 0.5f * (shape + transpose(shape));

        const float3 in_plane = wi - dot(wi, w) * w;
        const float3 t = dot(in_plane, in_plane) > 1e-12f ? normalize(in_plane) : u;
        const float3 b = cross(w, t);
        const float2 t2 = float2(dot(t, u), dot(t, v));
        const float2 b2 = float2(dot(b, u), dot(b, v));
        const float side = dot(wi, 3.f * n0 + dn1 + dn2) >= 0.f ? 1.f : -1.f;
        return side * float2(dot(b2, mul(symmetric, b2)), dot(t2, mul(symmetric, t2)));
    }

    Surface get_surface(const Hit hit, const TriangleHit tri, const bool prev_frame = false) {
        const float3 bary = hit.get_barycentrics3();
        const uint instance = hit.instance_index;

        const float3 pos = to_world_position(tri.gd, instance, tri.get_position(hit), prev_frame);
        const float3 edge01 =
            to_world_vector(tri.gd, instance, tri.get_triangle_edge01(), prev_frame);
        const float3 edge02 =
            to_world_vector(tri.gd, instance, tri.get_triangle_edge02(), prev_frame);

        float3 face_normal = normalize(cross(edge01, edge02));
        if (tri.gd.flags & GeometryDataFlags::FlipFacing)
            face_normal = -face_normal;

        float3 vertex_normal;
        if (tri.gd.flags & GeometryDataFlags::FlatShading) {
            vertex_normal = face_normal;
        } else {
            const float3 n_obj = normalize(bary.x * decode_normal(tri.v0.encoded_normal) +
                                           bary.y * decode_normal(tri.v1.encoded_normal) +
                                           bary.z * decode_normal(tri.v2.encoded_normal));
            vertex_normal = to_world_normal(tri.gd, instance, n_obj, prev_frame);
        }

        // per-vertex tangent or Lengyel UV-derived fallback
        float3 tangent;
        float handedness;
        bool tangent_valid;
        if (tri.gd.flags & GeometryDataFlags::HasTangents) {
            const float4 t0 = decode_tangent(tri.v0.encoded_tangent);
            const float4 t1 = decode_tangent(tri.v1.encoded_tangent);
            const float4 t2 = decode_tangent(tri.v2.encoded_tangent);
            const float4 t = bary.x * t0 + bary.y * t1 + bary.z * t2;
            tangent = to_world_vector(tri.gd, instance, normalize(t.xyz), prev_frame);
            handedness = sign(t.w);
            tangent_valid = true;
        } else {
            const float2 duv1 = float2(tri.v1.uv) - float2(tri.v0.uv);
            const float2 duv2 = float2(tri.v2.uv) - float2(tri.v0.uv);
            const float det = duv1.x * duv2.y - duv2.x * duv1.y;
            if (abs(det) > 1e-8f) {
                const float winding_sign =
                    (tri.gd.flags & GeometryDataFlags::FlipFacing) ? 1.f : -1.f;
                tangent = (duv2.y * edge01 - duv1.y * edge02) / det;
                handedness = sign(det) * winding_sign;
                tangent_valid = true;
            } else {
                tangent = float3(1, 0, 0);
                handedness = 1.f;
                tangent_valid = false;
            }
        }

        Surface s;
        s.material_id = tri.gd.material_id;
        s.pos = pos;
        s.uv = tri.get_uv(hit);
        s.face_normal = face_normal;
        s.spawn_offset = triangle_spawn_offset(pos, edge01, edge02);
        s.area = 0.5f * length(cross(edge01, edge02));
        s.frame = tangent_valid ? OrthonormalFrame(vertex_normal, tangent, handedness)
                                : OrthonormalFrame(vertex_normal);
        return s;
    }

    // a triangle inactive (x = NaN) in that frame stays where it is in the other
    float3 get_world_position(const Hit hit, const GeometryData gd, const bool prev_frame) {
        const float3 pos = _get_world_position(hit, gd, prev_frame);
        if (isnan(pos.x))
            return _get_world_position(hit, gd, !prev_frame);
        return pos;
    }

    float3 _get_world_position(const Hit hit, const GeometryData gd, const bool prev_frame) {
        const float3 bary = hit.get_barycentrics3();
        const uint3 indices = get_vertex_indices(gd, hit.primitive_id);
        const VertexBufferAddress vertices = get_vertices(gd, prev_frame);
        const float3 object_pos = bary.x * vertices[indices.x].position +
                                  bary.y * vertices[indices.y].position +
                                  bary.z * vertices[indices.z].position;
        return to_world_position(gd, hit.instance_index, object_pos, prev_frame);
    }

    // tri of the other frame, reused where its vertices did not change
    float3 get_world_position(const Hit hit, const TriangleHit tri, const bool prev_frame) {
        if (has_static_vertices(tri.gd))
            return to_world_position(tri.gd, hit.instance_index, tri.get_position(hit), prev_frame);
        return get_world_position(hit, tri.gd, prev_frame);
    }

    // Pixel-space delta of the surface point between the previous and the current projection.
    float2 compute_motion_vector(const float3 prev_pos,
                                 const float3 world_pos,
                                 const uint2 pixel,
                                 const uint2 dim) {
        if (merian_hint_enable_thin_lens && camera.aperture_radius > 0.f) {
            // use the center pixel in this case too
            return prev_camera.project_pinhole(prev_pos, dim) -
                   camera.project_pinhole(world_pos, dim);
        }
        return prev_camera.project_pinhole(prev_pos, dim) - float2(pixel) - camera.jitter;
    }

    // with_material = false => does not load surface material and keeps the interpolated frame
    // (no normal map)
    ShadingPoint get_shading_point(const RayDesc ray,
                                   const Hit hit,
                                   const GeometryData gd,
                                   const RenderContext ctx = RenderContext(),
                                   const LODSampler2D lod = DefaultLODSampler(),
                                   const bool with_material = true) {
        const TriangleHit tri = get_triangle_hit(hit, gd);
        return get_shading_point(ray, hit, tri, get_surface(hit, tri), ctx, lod, with_material);
    }

    // tri and surface of the frame of prev_frame
    ShadingPoint get_shading_point(const RayDesc ray,
                                   const Hit hit,
                                   const TriangleHit tri,
                                   const Surface surface,
                                   const RenderContext ctx = RenderContext(),
                                   const LODSampler2D lod = DefaultLODSampler(),
                                   const bool with_material = true,
                                   const bool prev_frame = false) {
        const float3 prev_pos = prev_frame ? surface.pos : get_world_position(hit, tri, true);

        OrthonormalFrame frame = surface.frame;
        float3 face_normal = surface.face_normal;
        bool front_facing = true;
        if (tri.gd.flags & GeometryDataFlags::TwoSided) {
            front_facing = dot(-ray.Direction, surface.face_normal) >= 0.f;
            if (!front_facing) {
                frame.flip_normal();
                face_normal = -face_normal;
            }
        }

        const ShadingData sd =
            ShadingData(-ray.Direction, surface.pos, surface.uv, get_time(prev_frame),
                        frame, face_normal, ctx.bounce, front_facing);

        ShadingPoint sp;
        sp.pos = surface.pos;
        sp.prev_pos = prev_pos;
        sp.enc_face_normal = encode_normal(surface.face_normal);
        sp.has_surface_material = with_material;
        sp.material_id = surface.material_id;
        sp.spawn_offset = surface.spawn_offset;
        sp.area = surface.area;
        if (with_material) {
            sp.surface_material = material_system.get_material_sample(
                surface.material_id, sd, sp.emission, sp.shading_frame, lod, prev_frame);
        } else {
            sp.emission = material_system.get_emission(surface.material_id, sd, lod, prev_frame);
            sp.shading_frame = frame;
        }
        return sp;
    }

    ShadingPoint get_shading_point(const RayDesc ray,
                                   const Hit hit,
                                   const RenderContext ctx = RenderContext(),
                                   const LODSampler2D lod = DefaultLODSampler(),
                                   const bool with_material = true,
                                   const bool prev_frame = false) {
        if (is_distant(hit))
            return _make_environment_map_shading_point(ray, ctx, lod, prev_frame);
        const TriangleHit tri = get_triangle_hit(hit, prev_frame);
        return get_shading_point(ray, hit, tri, get_surface(hit, tri, prev_frame), ctx, lod,
                                 with_material, prev_frame);
    }

    // record: the light cut learns from the draws.
    Optional<DirectLightSample> sample_direct_light<T : IDirectLightTarget>(
        const DirectLightVertex vertex,
        const T target,
        inout RandomGenerator rng,
        const bool record = true) {
        DirectLightReservoir reservoir = DirectLightReservoir();
        const uint draws = nee.draw_count();
        const LightCellWeights weights = nee.grid.weigh_cell(vertex, draws, rng);
        for (uint i = 0; i < draws; i++) {
            uint entry;
            bool delivers = false;
            if (let c = nee.draw_light(vertex, weights, i, rng, entry)) {
                if (target.receives(c.wo)) {
                    if (let ls = nee.light_sample(vertex, c, material_system, env_map, time)) {
                        const float light_pdf = nee.pdf_lights(vertex, c, weights);
                        float scatter_pdf;
                        const float3 unshadowed = target.unshadowed(ls, rng, scatter_pdf);
                        delivers = yuv_luminance(unshadowed) > 0.f;
                        reservoir.add_light(ls, c.pos, unshadowed, light_pdf, scatter_pdf, entry,
                                            rng);
                    }
                }
            }
            // only the winner is traced; a draw the receiver cannot see delivers nothing
            if (record && !delivers)
                nee.record_failed_draw(entry);
        }
        if (!reservoir.needs_shadow_ray()) {
            return none;
        }
        const float weight = reservoir.weight();
        DirectLightSample s;
        s.value = reservoir.value * weight;
        s.light = reservoir.light;
        s.light_point = reservoir.light_point;
        s.light_entry = reservoir.light_entry;
        s.light_pdf = reservoir.light_pdf;
        s.scatter_pdf = reservoir.scatter_pdf;
        s.realized_pdf = reservoir.realized_pdf();
        return s;
    }

    // One draw; pdf is NEE::mixture_pdf of it.
    Optional<LightSample> sample_direct_light(const DirectLightVertex vertex,
                                              inout RandomGenerator rng) {
        const LightCellWeights weights = nee.grid.weigh_cell(vertex, 1, rng);
        uint entry;
        if (let c = nee.draw_light(vertex, weights, 0, rng, entry)) {
            if (let ls = nee.light_sample(vertex, c, material_system, env_map, time)) {
                LightSample s = ls;
                s.pdf = nee.mixture_pdf(nee.pdf_lights(vertex, c, weights));
                if (s.pdf > 0.f)
                    return s;
            }
        }
        return none;
    }

    // Ray from the receiver (offset along its face normal) towards a point on a light. Distant
    // samples pass through sky portals (UseEnvMap geometry) to the environment behind them.
    RayDesc make_light_ray(const float3 pos,
                           const float3 face_normal,
                           const float spawn_offset,
                           const float3 light_pos,
                           const bool distant) {
        const float3 origin = offset_ray_origin(pos, face_normal, spawn_offset, light_pos - pos);
        // aim at the point itself: a ray along the unoffset direction that stops at the offset
        // distance clips the light triangle at grazing angles
        const float3 to_light = light_pos - origin;
        const float dist = length(to_light);

        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = dist > 0.f ? to_light / dist : float3(0, 0, 1);
        ray.TMin = SCENE_RAY_TMIN;
        ray.TMax = distant || !(dist > 0.f) ? SCENE_RAY_TMAX : dist;
        return ray;
    }

    bool has_sky_portals(const bool prev_frame) {
        return prev_frame ? prev_nee.has_sky_portals : nee.has_sky_portals;
    }

    uint light_ray_flags(const bool distant, const bool prev_frame) {
        // a sky portal in front of the environment must not end the search
        return distant && has_sky_portals(prev_frame) ? RAY_FLAG_NONE
                                                      : RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    }

    bool reaches_light(const Hit hit, const bool distant, const bool prev_frame) {
        if (!hit.is_valid())
            return true;
        return distant && has_sky_portals(prev_frame) && is_distant(hit);
    }

    bool trace_light_visibility(const float3 pos,
                                const float3 face_normal,
                                const float spawn_offset,
                                const float3 light_pos,
                                const bool distant,
                                const uint instances_mask = 0xff,
                                const bool prev_frame = false) {
        const RayDesc ray = make_light_ray(pos, face_normal, spawn_offset, light_pos, distant);
        return reaches_light(
            trace_ray(ray, light_ray_flags(distant, prev_frame), instances_mask, prev_frame),
            distant, prev_frame);
    }

    // the light cut learns from the outcome
    bool trace_direct_light(const DirectLightVertex vertex,
                            const DirectLightSample s,
                            inout RandomGenerator rng,
                            const uint instances_mask = 0xff) {
        const float radiance = yuv_luminance(s.light.Li);
        const bool visible =
            trace_light_visibility(vertex.pos, vertex.face_normal, vertex.spawn_offset, s.light.pos,
                                   s.light.index == LIGHT_ENV_INDEX, instances_mask);
        nee.record_contribution(vertex, s.light.index, s.light.wo, visible ? radiance : 0.f,
                                s.light.dist, rng, s.light_entry);
        return visible;
    }

    float3 resolve_direct_light(const DirectLightVertex vertex,
                                const Optional<DirectLightSample> direct,
                                inout RandomGenerator rng,
                                const uint instances_mask = 0xff) {
        if (let s = direct)
            return trace_direct_light(vertex, s, rng, instances_mask) ? s.value : float3(0);
        return float3(0);
    }

    static float3
    get_receiver_normal(const float3 incident, const float3 face_normal, const bool transmissive) {
        if (transmissive)
            return float3(0);
        return dot(face_normal, incident) < 0.f ? face_normal : -face_normal;
    }

    float4 debug_direct_light(const RayDesc ray,
                              const ShadingPoint sp,
                              const uint instance_mask,
                              inout RandomGenerator rng) {
        if (!sp.has_surface_material || !nee.has_lights())
            return float4(0);
        const float3 ng = sp.get_face_normal();
        const float3 normal = get_receiver_normal(
            ray.Direction, ng,
            bsdf_is_transmissive(sp.surface_material.get_bsdf().get_lobe_types()));
        let vertex = nee.begin_vertex(sp.pos, ng, sp.spawn_offset, normal, rng);

        switch (nee.debug_view) {
        case LightDebugView::LightDebugCell:
            return vertex.has_cell() ? float4(random_color(vertex.cell), 1) : float4(0, 0, 0, 1);
        case LightDebugView::LightDebugSlots: {
            if (!vertex.has_cell())
                return float4(0, 0, 0, 1);
            const LightCellWeights weights = nee.grid.weigh_cell(vertex);
            float listed = 0.f;
            float learned = 0.f;
            float actual = 0.f;
            const uint base = light_bucket_base(vertex.cell);
            const float inv_cell_scale = 1.f / nee.grid.cell_scale(vertex.cell);
            for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
                const uint2 range = nee.grid.slot_range(base + slot);
                if (range.x >= range.y)
                    continue;
                listed += 1.f;
                const float p = nee.grid.slot_share(vertex, base + slot, weights);
                learned += p * nee.grid.contribution[base + slot].x *
                           nee.grid.estimate[base + slot] * inv_cell_scale;
                uint texel;
                float share_in_slot;
                const uint drawn = nee.draw_in_slot(base + slot, rng, texel, share_in_slot);
                let lp = nee.light_point(vertex, drawn, texel, base + slot, rng);
                if (!lp.hasValue)
                    continue;
                const bool distant = drawn == LIGHT_ENV_INDEX;
                RayDesc probe = make_light_ray(sp.pos, ng, sp.spawn_offset, lp.value.pos, distant);
                probe.TMax = SCENE_RAY_TMAX;
                const uint hit = get_light_index(trace_ray(probe, RAY_FLAG_NONE, instance_mask));
                const uint position = nee.tree_position(hit, probe.Direction);
                if (position >= range.x && position < range.y)
                    actual += p;
            }
            return float4(listed / float(LIGHT_GRID_SLOTS), learned, actual, 1);
        }
        case LightDebugView::LightDebugCoverage: {
            const OrthonormalFrame frame = OrthonormalFrame(any(normal != 0.f) ? normal : ng);
            float3 found = float3(0);
            for (uint i = 0; i < LIGHT_DEBUG_COVERAGE_RAYS; i++) {
                RayDesc probe;
                probe.Direction = frame.to_world(merian::sample_cos(rng));
                probe.Origin = offset_ray_origin(sp.pos, ng, sp.spawn_offset, probe.Direction);
                probe.TMin = SCENE_RAY_TMIN;
                probe.TMax = SCENE_RAY_TMAX;
                const Hit hit = trace_ray(probe, RAY_FLAG_NONE, instance_mask);
                const float emitted = yuv_luminance(
                    get_shading_point(probe, hit, RenderContext(), DefaultLODSampler(), false)
                        .emission);
                if (!(emitted > 0.f))
                    continue;
                const uint index = get_light_index(hit);
                if (index == LIGHT_INVALID_INDEX)
                    found.b += emitted;
                else if (nee.pdf(vertex, index, probe.Origin, probe.Direction) > 0.f)
                    found.r += emitted;
                else if (index == LIGHT_ENV_INDEX ||
                         (nee.table.triangles[index].flux > 0.f &&
                          nee.pdf_triangle(index, probe.Origin, probe.Direction) > 0.f))
                    found.g += emitted;
                else
                    found.b += emitted;
            }
            const float total = found.r + found.g + found.b;
            return total > 0.f ? float4(found / total, 1) : float4(0, 0, 0, 1);
        }
        default: {
            float3 outcome = float3(0);
            for (uint i = 0; i < LIGHT_DEBUG_DRAWS; i++) {
                let lp = nee.sample_point(vertex, rng);
                if (!lp.hasValue)
                    continue;
                const bool distant = lp.value.index == LIGHT_ENV_INDEX;
                RayDesc probe = make_light_ray(sp.pos, ng, sp.spawn_offset, lp.value.pos, distant);
                if (any(normal != 0.f) && dot(probe.Direction, normal) <= 0.f)
                    continue;
                const float light_dist = probe.TMax;
                probe.TMax = SCENE_RAY_TMAX;
                const uint hit = get_light_index(trace_ray(probe, RAY_FLAG_NONE, instance_mask));
                const bool reached = distant ? hit == LIGHT_ENV_INDEX
                                             : nee.reaches(lp.value.index, lp.value.entry, hit);
                if (reached) {
                    outcome.g += 1.f;
                } else {
                    probe.TMax = 0.999f * light_dist;
                    if (distant || trace_ray(probe, RAY_FLAG_NONE, instance_mask).is_valid())
                        outcome.r += 1.f;
                    else
                        outcome.b += 1.f;
                }
            }
            return float4(outcome / float(LIGHT_DEBUG_DRAWS), 1);
        }
        }
    }

    // LIGHT_ENV_INDEX, LIGHT_INVALID_INDEX for a non-emitter; with prev_frame in prev_nee
    uint get_light_index(const Hit hit, const bool prev_frame = false) {
        if (is_distant(hit))
            return LIGHT_ENV_INDEX;
        if (!prev_frame &&
            (get_geometry_data(hit.get_geometry_id()).flags & GeometryDataFlags::Emissive) == 0)
            return LIGHT_INVALID_INDEX;
        const Optional<uint> index =
            prev_frame ? prev_nee.get_triangle_index(hit.get_geometry_id(), hit.primitive_id)
                       : nee.get_triangle_index(hit.get_geometry_id(), hit.primitive_id);
        return index.hasValue ? index.value : LIGHT_INVALID_INDEX;
    }

    Hit get_emitter_hit(const uint index, const float3 pos) {
        const EmissiveTriangle tri = nee.table.triangles[index];
        const LightGeometry lg = nee.table.geometries[find_light_geometry(
            nee.table.geometries, nee.table.geometry_count, index)];
        float2 bary = tri.get_barycentrics(pos);
        // the table winds every triangle front-facing, the mesh does not
        if ((get_geometry_data(tri.geometry_id).flags & GeometryDataFlags::FlipFacing) != 0)
            bary = bary.yx;
        return Hit(tri.geometry_id, lg.instance_index, tri.primitive_id, 0, bary);
    }

    float3 eval_env_map(const ShadingData sd,
                        const LODSampler2D lod = DefaultLODSampler(),
                        const bool prev_frame = false) {
        const Textures textures = material_system.texture_manager.get_textures(prev_frame);
        if (prev_frame)
            return prev_env_map.eval(sd, textures, lod);
        return env_map.eval(sd, textures, lod);
    }

    // at the time of the frame of `prev_frame`
    float3 eval_env_map(const float3 dir, const float3 origin, const bool prev_frame) {
        return eval_env_map(make_env_shading_data(dir, origin, get_time(prev_frame)),
                            DefaultLODSampler(), prev_frame);
    }

    ShadingPoint _make_environment_map_shading_point(const RayDesc ray,
                                                     const RenderContext ctx,
                                                     const LODSampler2D lod,
                                                     const bool prev_frame = false) {
        const float3 pos = ray.Origin + ray.Direction * ENV_MAP_DISTANCE;
        const ShadingData sd =
            ShadingData(-ray.Direction, pos, float2(0), get_time(prev_frame),
                        OrthonormalFrame(-ray.Direction), -ray.Direction, ctx.bounce, true);

        ShadingPoint sp;
        sp.pos = pos;
        sp.prev_pos = pos;
        sp.enc_face_normal = encode_normal(-ray.Direction);
        sp.emission = eval_env_map(sd, lod, prev_frame);
        sp.shading_frame = OrthonormalFrame(-ray.Direction);
        sp.has_surface_material = false;
        sp.material_id = uint(-1);
        sp.spawn_offset = 0.f;
        sp.area = 0.f;
        return sp;
    }

    void _proceed_ray(inout SceneAccelerationStructure::RayQueryImpl rq,
                      const bool prev_frame = false) {
        while (rq.Proceed()) {

            if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                Hit candidate = Hit(rq.CandidateInstanceID(), rq.CandidateInstanceIndex(),
                                    rq.CandidatePrimitiveIndex(), rq.CandidateGeometryIndex(),
                                    rq.CandidateRayBarycentrics());
                if (prev_frame)
                    candidate = from_prev_frame(candidate);
                const GeometryData geometry_data = get_geometry_data(candidate.get_geometry_id());
                if (material_system.alpha_test(
                        geometry_data.material_id,
                        get_interpolated_uv(geometry_data, candidate.primitive_id,
                                            candidate.get_barycentrics3(), prev_frame),
                        DefaultLODSampler(), prev_frame)) {
                    continue;
                }

                rq.CommitNonOpaqueTriangleHit();
            }
        }
    }
}

} // namespace merian
