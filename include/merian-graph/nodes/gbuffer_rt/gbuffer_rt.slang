import merian_shaders.scene.scene;
import merian_shaders.gbuffer;
import merian_shaders.utils.textures;
import merian_shaders.utils.ray_differential;
import merian_shaders.colors.debug;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;

namespace merian {

RWTexture2D<float4> emission;
// 8-bit TLAS instance cull mask used by the primary ray query. Per-frame from the host.
struct GBufferParams {
    uint instance_mask;
};
ConstantBuffer<GBufferParams> params;

extern static const bool merian_gbuffer_write_emission = true;
extern static const bool merian_gbuffer_write_specular_hit_distance = false;
// 0 samples, 1 reflects, 2 transmits, 3 follows Fresnel
extern static const int merian_gbuffer_specular_lobe = 3;
extern static const float merian_gbuffer_specular_max_roughness = 0.25;
extern static const int merian_gbuffer_specular_max_bounces = 2;
extern static const bool merian_gbuffer_specular_psr = false;

// exp2 of the texture LOD bias
extern static const float merian_gbuffer_texture_lod_scale = 1.0;

// A fixed sample decides the lobe of a delta BSDF.
struct ConstantRandom : RandomGenerator {
    float value;

    typealias CloneType = ConstantRandom;

    CloneType clone() {
        return this;
    }

    [mutating]
    uint next() {
        return uint(value * 4294967295.0);
    }

    [mutating]
    override float next_float() {
        return value;
    }
}

// The surface the chain of smooth surfaces ends on. `travelled` is 0 and `sp` the one it started
// from where that surface is rough already.
struct SpecularChain {
    ShadingPoint sp;
    RayDesc ray;
    float travelled;
}

SpecularChain trace_specular_chain(const ParameterBlock<Scene> scene,
                                   const uint instance_mask,
                                   RayDesc ray,
                                   ShadingPoint sp,
                                   inout RandomGenerator rng) {
    float travelled = 0.f;
    for (int bounce = 0; bounce < merian_gbuffer_specular_max_bounces; bounce++) {
        if (!sp.has_surface_material) {
            return {sp, ray, bounce == 0 ? 0.f : MAX_LINEAR_Z};
        }
        let bsdf = sp.surface_material.get_bsdf();
        if (sqrt(bsdf.specular_alpha) >= merian_gbuffer_specular_max_roughness) {
            return {sp, ray, travelled};
        }

        const OrthonormalFrame frame = sp.shading_frame;
        const float3 wi_local = frame.to_local(-ray.Direction);
        const Optional<float3> wo_local = bsdf.sample(wi_local, rng);
        if (!wo_local.hasValue) {
            return {sp, ray, travelled};
        }
        const Optional<RayDesc> next = sp.make_ray(-ray.Direction, frame.to_world(wo_local.value),
                                                   wi_local.z * wo_local.value.z < 0.f);
        if (!next.hasValue) {
            return {sp, ray, travelled};
        }

        const float3 from = sp.pos;
        ray = next.value;
        sp = scene.get_shading_point(ray, scene.trace_ray(ray, RAY_FLAG_NONE, instance_mask),
                                     RenderContext(uint16_t(bounce + 1)), DefaultLODSampler(), true);
        travelled += distance(from, sp.pos);
    }
    return {sp, ray, min(travelled, MAX_LINEAR_Z)};
}

SpecularChain specular_chain(const ParameterBlock<Scene> scene,
                             const uint instance_mask,
                             const RayDesc ray,
                             const ShadingPoint sp,
                             const uint2 pixel) {
    if (merian_gbuffer_specular_lobe == 0) {
        RandomGenerator rng = XorShift32(pcg3d16(uint3(pixel, scene.frame)));
        return trace_specular_chain(scene, instance_mask, ray, sp, rng);
    }
    RandomGenerator rng = ConstantRandom(merian_gbuffer_specular_lobe == 1   ? 0.f
                                         : merian_gbuffer_specular_lobe == 2 ? 1.f
                                                                             : 0.5f);
    return trace_specular_chain(scene, instance_mask, ray, sp, rng);
}

GBufferSample make_sample(const Camera camera,
                          const float3 pos,
                          const float3 normal,
                          const float linear_z,
                          const half2 grad_z,
                          const float delta_z,
                          const float2 motion_vectors,
                          const Hit hit) {
    GBufferSample sample;
    sample.normal = normal;
    sample.linear_z = min(linear_z, MAX_LINEAR_Z);
    sample.grad_z = grad_z;
    sample.delta_z = delta_z;
    sample.motion_vectors = motion_vectors;
    sample.hit = hit;
    sample.view_depth = min(dot(pos - camera.position, normalize(camera.W)), MAX_LINEAR_Z);
    sample.projected_depth = (camera.far / (camera.far - camera.near)) *
                             (1.f - camera.near / max(sample.view_depth, 1e-6f));
    return sample;
}

[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<WGBuffer> gbuffer) {
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 dim = gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    const RayDesc ray = scene.generate_camera_ray(pixel, dim);
    const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, params.instance_mask);

    if (let gd = scene.get_geometry_data(hit)) {
        if ((gd.flags & GeometryDataFlags::UseEnvMap) == 0) {
            const TriangleHit tri = scene.get_triangle_hit(hit, gd);
            const Surface surface = scene.get_surface(hit, tri);
            const float3 prev_pos = scene.get_prev_world_position(hit, tri);

            const float linear_z = length(surface.pos - ray.Origin);
            const float3 r_x = scene.camera.get_pinhole_direction(pixel + uint2(1, 0), dim);
            const float3 r_y = scene.camera.get_pinhole_direction(pixel + uint2(0, 1), dim);
            const half2 grad_z = surface.compute_grad_z(ray, r_x, r_y, linear_z);
            const float delta_z = length(prev_pos - scene.prev_camera.position) - linear_z;
            const float2 mv = scene.compute_motion_vector(prev_pos, surface.pos, pixel, dim);

            // Screen-space UV gradients for anisotropic mip selection.
            const float3 edge01_w =
                scene.to_world_vector(gd, hit.instance_index, tri.get_triangle_edge01());
            const float3 edge02_w =
                scene.to_world_vector(gd, hit.instance_index, tri.get_triangle_edge02());
            RayDifferential rd = RayDifferential::from_pinhole_camera(ray.Direction, r_x, r_y);
            rd.propagate(ray.Direction, linear_z, surface.face_normal);
            float2 ddx, ddy;
            rd.compute_uv_gradients(edge01_w, edge02_w, surface.face_normal, tri.v0.uv, tri.v1.uv,
                                    tri.v2.uv, ddx, ddy);
            ddx *= merian_gbuffer_texture_lod_scale;
            ddy *= merian_gbuffer_texture_lod_scale;

            OrthonormalFrame frame = surface.frame;
            float3 face_normal = surface.face_normal;
            bool front_facing = true;
            if (gd.flags & GeometryDataFlags::TwoSided) {
                front_facing = dot(-ray.Direction, surface.face_normal) >= 0.f;
                if (!front_facing) {
                    frame.flip_normal();
                    face_normal = -face_normal;
                }
            }

            const ShadingData sd = ShadingData(-ray.Direction, surface.pos, surface.uv, scene.time,
                                               frame, face_normal, uint16_t(0), front_facing);
            float3 mat_emission;
            OrthonormalFrame mat_shading_frame;
            let mat = scene.material_system.get_material_sample(
                surface.material_id, sd, mat_emission, mat_shading_frame, GradLODSampler(ddx, ddy));

            let bsdf = mat.get_bsdf();
            const Albedo albedo = bsdf.get_albedo(mat_shading_frame.to_local(-ray.Direction));
            GBufferSample sample = make_sample(scene.camera, surface.pos,
                                               mat_shading_frame.get_normal(), linear_z, grad_z,
                                               delta_z, mv, hit);
            sample.albedo = albedo.reflection;
            sample.diffuse_albedo = albedo.diffuse;
            sample.specular_albedo = albedo.specular;
            sample.roughness = sqrt(bsdf.specular_alpha);
            if (merian_gbuffer_write_specular_hit_distance || merian_gbuffer_specular_psr) {
                const SpecularChain chain = specular_chain(
                    scene, params.instance_mask, ray,
                    scene.get_shading_point(ray, hit, RenderContext(), GradLODSampler(ddx, ddy)),
                    pixel);
                sample.specular_hit_distance = chain.travelled;

                if (merian_gbuffer_specular_psr && chain.travelled > 0.f) {
                    const float virtual_z = linear_z + chain.travelled;
                    const float3 virtual_pos = ray.Origin + virtual_z * ray.Direction;
                    if (chain.sp.has_surface_material) {
                        let chain_bsdf = chain.sp.surface_material.get_bsdf();
                        const OrthonormalFrame chain_frame = chain.sp.shading_frame;
                        const Albedo chain_albedo =
                            chain_bsdf.get_albedo(chain_frame.to_local(-chain.ray.Direction));
                        sample = make_sample(scene.camera, virtual_pos, chain_frame.get_normal(),
                                             virtual_z, grad_z, delta_z, mv, hit);
                        sample.albedo = chain_albedo.reflection;
                        sample.diffuse_albedo = chain_albedo.diffuse;
                        sample.specular_albedo = chain_albedo.specular;
                        sample.roughness = sqrt(chain_bsdf.specular_alpha);
                    } else {
                        sample = make_sample(scene.camera, virtual_pos, chain.sp.shading_normal,
                                             virtual_z, grad_z, delta_z, mv, hit);
                        sample.albedo = chain.sp.emission;
                        sample.diffuse_albedo = float3(0.5f);
                        sample.specular_albedo = float3(0.f);
                        sample.roughness = 0.f;
                    }
                    // the distance starts at the surface the guides describe
                    sample.specular_hit_distance = 0.f;
                }
            }
            gbuffer.store(pixel, sample);
            // The emission output bypasses the renderer, so it carries the attenuation of the
            // primary segment itself. The albedo stays a pure material property.
            if (merian_gbuffer_write_emission)
                emission[pixel] =
                    float4(mat_emission * scene.exterior_volume.transmittance(linear_z), 0);
            return;
        }
    }

    const ShadingPoint sp =
        scene._make_environment_map_shading_point(ray, RenderContext(), DefaultLODSampler());
    const float linear_z = length(sp.pos - ray.Origin);
    const float2 mv = scene.compute_motion_vector(sp.pos, sp.pos, pixel, dim);

    GBufferSample sample =
        make_sample(scene.camera, sp.pos, sp.shading_normal, linear_z, half2(0), 0, mv, hit);
    sample.albedo = sp.emission;
    // sky defaults a demodulating denoiser expects (DLSS-RR Integration Guide 3.4.1, 3.4.2)
    sample.diffuse_albedo = float3(0.5f);
    sample.specular_albedo = float3(0.f);
    sample.roughness = 0.f;
    sample.specular_hit_distance = MAX_LINEAR_Z;
    gbuffer.store(pixel, sample);
    if (merian_gbuffer_write_emission)
        emission[pixel] = float4(sp.emission * scene.exterior_volume.transmittance(linear_z), 0);
}

} // namespace merian
