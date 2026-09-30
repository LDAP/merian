import merian_shaders.debug.path_record;
import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.colors.colorspaces;
__exported import merian_shaders.sampling.guiding;
__exported import render_pt_guiding;

using merian;

namespace merian {
static const int OUTPUT_VIEW_RADIANCE = 0;
static const int OUTPUT_VIEW_GUIDING_DEBUG = 1;
static const int OUTPUT_VIEW_SCATTER_STATISTICS = 2;

extern static const bool merian_render_emission_on_primary = true;
extern static const int merian_render_output_view = OUTPUT_VIEW_RADIANCE;
extern static const int merian_render_spp = 1;
extern static const uint merian_render_seed = 0;
extern static const int merian_render_max_path_length = 8;
extern static const uint merian_render_instance_mask = 0xFFu;
extern static const bool merian_render_enable_ser = false;
extern static const bool merian_render_demodulate_albedo = false;
extern static const bool merian_render_follow_specular = true;
extern static const float merian_render_specular_alpha = 0.f;
extern static const int merian_render_nee_mode = NEE_MODE_RESAMPLED;
extern static const float merian_render_nee_probability = 0.5;
extern static const int merian_render_nee_bounces = 0;
extern static const int merian_render_scatter_mode = SCATTER_MODE_MIS;
extern static const int merian_render_scatter_candidates = 2;
extern static const bool merian_render_russian_roulette = true;
} // namespace merian

interface IThreadReorder {
    static void reorder();
}

struct NoThreadReorder : IThreadReorder {
    static void reorder() {}
}

struct SerThreadReorder : IThreadReorder {
    static void reorder() {
#if defined(MERIAN_SPIRV_CAP_SUPPORTED_ShaderInvocationReorderEXT)
        if (merian_render_enable_ser) {
            ReorderThread(0, 0);
        }
#endif
    }
}

struct RenderBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
    ParameterBlock<RenderGuiding> guiding;
    RWStructuredBuffer<uint, ScalarDataLayout> path_records;
}

struct ScatterTarget<B : ShadingFunction, G : GuidingDistribution<float3>, V : HomogeneousVolume>
    : IDirectLightTarget {
    B bsdf;
    G guiding;
    ScatterMixture mixture;
    PathMedium<V> medium;
    OrthonormalFrame frame;
    float3 wi_local;
    float3 shading_normal;
    float3 face_normal;

    float3 unshadowed(const LightSample ls, inout RandomGenerator rng) {
        const float3 ng = dot(face_normal, shading_normal) < 0.f ? -face_normal : face_normal;
        if ((dot(ls.wo, shading_normal) < 0.f) != (dot(ls.wo, ng) < 0.f)) {
            return float3(0);
        }
        return bsdf.eval(wi_local, frame.to_local(ls.wo), rng) * ls.Li *
               medium.transmittance(ls.dist);
    }

    float scatter_pdf(const LightSample ls) {
        return mixture.pdf(bsdf, guiding, wi_local, frame.to_local(ls.wo), ls.wo);
    }
}

ScatterTarget<B, G, V>
make_scatter_target<B : ShadingFunction, G : GuidingDistribution<float3>, V : HomogeneousVolume>(
    const B bsdf,
    const G guiding,
    const ScatterMixture mixture,
    const PathMedium<V> medium,
    const OrthonormalFrame frame,
    const float3 wi_local,
    const float3 shading_normal,
    const float3 face_normal) {
    ScatterTarget<B, G, V> target = {bsdf,  guiding,  mixture,        medium,
                                     frame, wi_local, shading_normal, face_normal};
    return target;
}

void record_connection(inout PathRecorder rec,
                       const ParameterBlock<RenderBinding> params,
                       const float3 from,
                       const bool attempted,
                       const Optional<DirectLightSample> direct,
                       const float3 throughput,
                       const float3 contrib,
                       const uint bounce,
                       const uint material) {
    if (let d = direct) {
        rec.connection(params.path_records, from, d.light_pos, d.realized_pdf, throughput * d.value,
                       contrib, bounce, PATH_RECORD_METHOD_LIGHT, material);
    } else if (attempted) {
        rec.connection(params.path_records, from, from, 0.f, float3(0.f), float3(0.f), bounce,
                       PATH_RECORD_METHOD_LIGHT, material);
    }
}

float3 trace_path<Reorder : IThreadReorder>(const ParameterBlock<Scene> scene,
                                            const ParameterBlock<RenderBinding> params,
                                            const uint2 pixel,
                                            RayDesc ray,
                                            ShadingPoint sp,
                                            const uint16_t sample_index,
                                            inout RandomGenerator rng,
                                            inout float3 scatter_stats,
                                            inout PathRecorder rec) {
    var medium = scene.make_path_medium();

    float3 throughput = medium.transmittance(distance(ray.Origin, sp.pos));
    float3 radiance = merian_render_emission_on_primary ? throughput * sp.emission : float3(0);
    rec.emission(radiance);

    const RenderFrameInfo frame_info = {scene.camera.position, scene.prev_camera.position,
                                        scene.time, scene.time_diff, scene.frame};
    var guiding_state = params.guiding.begin_path();

    const bool nee = merian_render_nee_mode != NEE_MODE_OFF && scene.nee.has_lights();

    for (uint16_t bounce = 1; bounce < merian_render_max_path_length; bounce++) {
        if (!sp.has_surface_material || all(throughput < SCENE_THROUGHPUT_THRESHOLD)) {
            break;
        }

        const bool nee_here =
            nee && (merian_render_nee_bounces <= 0 || bounce <= merian_render_nee_bounces);
        const float3 ng = sp.get_face_normal();
        const float spawn_offset = sp.spawn_offset;
        let lights = scene.nee.begin_vertex(sp.pos, rng);
        // drawn before the guiding candidates exist, so the triangle record is not live with them
        const bool light_technique = merian_render_nee_mode == NEE_MODE_MIXTURE && nee_here &&
                                     rng.next_float() < merian_render_nee_probability;
        uint sampled_light = LIGHT_INVALID_INDEX;
        float3 light_dir = float3(0, 0, 1);
        if (light_technique) {
            if (let lp = scene.nee.sample_point(lights, rng)) {
                // aimed from the offset origin: from sp.pos it misses the triangle near its edges
                const float3 origin = offset_ray_origin(sp.pos, ng, spawn_offset, lp.pos - sp.pos);
                light_dir = normalize(lp.pos - origin);
                sampled_light = lp.index;
            }
        }

        const OrthonormalFrame frame = sp.shading_frame;
        let bsdf = sp.surface_material.get_bsdf();
        const float3 wi_local = frame.to_local(-ray.Direction);
        const float3 ns = frame.get_normal();
        const PathVertex vertex = {sp.pos,
                                   sp.prev_pos,
                                   frame,
                                   bsdf.effective_alpha,
                                   bounce,
                                   sample_index,
                                   pixel,
                                   false,
                                   dot(ng, ns) < 0.f ? -ng : ng,
                                   bsdf_is_transmissive(bsdf.get_lobe_types())};

        // 1. guiding
        let guiding = params.guiding.get_distribution(guiding_state, frame_info, vertex, rng);
        const ScatterMixture mixture =
            ScatterMixture(merian::guiding_share(guiding, bsdf.effective_alpha));

        const bool nee_surface = nee_here && bsdf.effective_alpha >= NEE_MIN_ALPHA;

        // 2. resampled direct light
        Optional<DirectLightSample> direct = none;
        if (merian_render_nee_mode == NEE_MODE_RESAMPLED && nee_surface) {
            direct = scene.sample_direct_light(
                lights,
                make_scatter_target(bsdf, guiding, mixture, medium, frame, wi_local, ns, ng),
                rng);
        }

        if (merian_render_russian_roulette) {
            const float survive = min(1.f, yuv_luminance(throughput));
            if (rng.next_float() >= survive)
                break;
            throughput /= survive;
        }

        // 3. scatter direction, from a light or from the mixture. The light technique's density
        // joins the mixture once the hit says which light the ray reached.
        const float p_light = merian_render_nee_mode == NEE_MODE_MIXTURE && nee_surface
                                  ? merian_render_nee_probability
                                  : 0.f;
        const bool light_scatter = light_technique && p_light > 0.f;
        bool scattered = false;
        bool guided = false;
        BSDFLobe scatter_lobe = BSDFLobe::None;
        float3 wo_world = float3(0, 0, 1);
        float3 f_cos = float3(0);
        float scatter_pdf = 0.f;
        float realized_pdf = 0.f;
        bool delta = false;
        if (light_scatter) {
            scattered = sampled_light != LIGHT_INVALID_INDEX;
            wo_world = light_dir;
            if (scattered) {
                const float3 wo_l = frame.to_local(wo_world);
                f_cos = bsdf.eval(wi_local, wo_l, rng);
                scatter_pdf = mixture.pdf(bsdf, guiding, wi_local, wo_l, wo_world);
                realized_pdf = scatter_pdf;
            }
        } else {
            sampled_light = LIGHT_INVALID_INDEX;
            const Optional<ScatterSample> scatter =
                merian_render_scatter_mode == SCATTER_MODE_RIS
                    ? mixture.sample_ris(bsdf, guiding, frame, merian_render_scatter_candidates,
                                         wi_local, rng)
                    : mixture.sample(bsdf, guiding, frame, wi_local, rng);
            if (let s = scatter) {
                wo_world = s.wo;
                f_cos = s.f_cos;
                scatter_pdf = s.pdf;
                realized_pdf = s.realized_pdf;
                delta = s.delta;
                guided = s.guided;
                scatter_lobe = s.lobe;
                scattered = true;
            }
        }
        const float3 wo_local = frame.to_local(wo_world);

        PathSegment segment;
        segment.pos = vertex.pos;
        segment.normal = frame.get_normal();
        segment.bounce = bounce;
        segment.wo = wo_world;
        segment.pdf = delta ? scatter_pdf : (1.f - p_light) * scatter_pdf;
        segment.weight = float3(0);
        segment.delta = delta;
        segment.traced = false;
        segment.is_last = bounce + uint16_t(1) >= merian_render_max_path_length;
        segment.next_pos = vertex.pos;
        segment.aim_pos = vertex.pos;
        segment.next_normal = segment.normal;
        segment.next_mv = half3(0);
        segment.incident = float3(0);
        segment.direct_scale = guiding_direct_scale(1.f);

        // taken after the mixture densities die, so only the record crosses the traversals below
        let guiding_record = params.guiding.record_scatter(guiding, segment, guided, rng);

        const float3 direct_light =
            throughput * scene.resolve_direct_light(sp.pos, ng, spawn_offset, direct,
                                                    merian_render_instance_mask);
        radiance += direct_light;

        Reorder.reorder();

        const bool transmission = wi_local.z * wo_local.z < 0.f;

        const float record_pdf = delta ? realized_pdf : (1.f - p_light) * realized_pdf;
        const float3 record_f_cos = throughput * f_cos;
        const uint record_lobe = path_record_pack_lobe(scatter_lobe) |
                                 (transmission ? PATH_RECORD_LOBE_TRANSMISSION : 0u);
        const uint record_method = light_scatter ? PATH_RECORD_METHOD_LIGHT
                                   : guided      ? PATH_RECORD_METHOD_GUIDING
                                                 : PATH_RECORD_METHOD_BSDF;
        const uint record_material = path_record_pack_material(sp.material_id);
        bool record_waits = p_light > 0.f && scattered && !delta;
        if (!record_waits) {
            rec.vertex(params.path_records, vertex.pos, wo_world, record_pdf,
                       segment.pdf > 0.f ? record_f_cos / segment.pdf : float3(0),
                       uint(bounce) - 1u, record_lobe, record_method, record_material);
            record_connection(rec, params, vertex.pos,
                              merian_render_nee_mode == NEE_MODE_RESAMPLED && nee_surface, direct,
                              throughput, direct_light,
                              uint(bounce) - 1u, record_material);
        }

        Optional<RayDesc> next_ray = none;
        if (scattered && any(f_cos > 0.f)) {
            next_ray = sp.make_ray(-ray.Direction, wo_world, transmission);
        }

        if (merian_render_output_view == OUTPUT_VIEW_SCATTER_STATISTICS && guided) {
            scatter_stats += float3(1.f, select(all(f_cos <= 0.f), 1.f, 0.f),
                                    select(any(f_cos > 0.f) && !next_ray.hasValue, 1.f, 0.f));
        }

        if (let next = next_ray) {
            if (merian_material_system_any_interior_volume && transmission) {
                medium.cross_boundary(sp.surface_material.get_interior_volume(),
                                      dot(-ray.Direction, ng) >= 0.f);
            }
            ray = next;

            float mis_weight = 1.f;
            float3 chain_bsdf = float3(1);
            float3 chain_transmittance = float3(1);
            float apparent_distance = 0.f;
            float apparent_depth_scale = 1.f;
            float mirror_focal_length = 0.f;
            bool through_specular = false;

            for (;;) {
                const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, merian_render_instance_mask);

                // before the hit is shaded; no light sampler reaches past a singular surface
                if (!through_specular && nee_surface && !delta) {
                    const uint light_index = scene.get_light_index(hit);
                    const float light_hit_pdf =
                        scene.nee.pdf(lights, light_index, ray.Origin, ray.Direction);
                    if (merian_render_nee_mode == NEE_MODE_RESAMPLED) {
                        mis_weight = segment.pdf / (light_hit_pdf + segment.pdf);
                    } else {
                        const float light_pdf = p_light * scene.nee.mixture_pdf(light_hit_pdf);
                        segment.pdf += light_pdf;
                        if (record_waits) {
                            rec.vertex(
                                params.path_records, vertex.pos, wo_world, record_pdf + light_pdf,
                                segment.pdf > 0.f ? record_f_cos / segment.pdf : float3(0),
                                uint(bounce) - 1u, record_lobe, record_method, record_material);
                            record_waits = false;
                        }
                        // occluded
                        if (sampled_light != LIGHT_INVALID_INDEX && sampled_light != light_index) {
                            segment.pdf = 0.f;
                        }
                    }
                }

                sp = scene.get_shading_point(ray, hit, RenderContext(bounce), DefaultLODSampler(),
                                             !segment.is_last);

                const float distance_travelled = distance(ray.Origin, sp.pos);
                chain_transmittance *= medium.transmittance(distance_travelled);
                // A curved mirror puts the image at its focal length R/2. One focusing in front
                // of itself has no virtual image and keeps the distance.
                float segment_length = distance_travelled;
                if (mirror_focal_length != 0.f) {
                    const float denominator = segment_length + mirror_focal_length;
                    if (denominator > 1e-4f) {
                        segment_length *= mirror_focal_length / denominator;
                    }
                    mirror_focal_length = 0.f;
                }
                apparent_distance += apparent_depth_scale * segment_length;

                if (!merian_render_follow_specular || !sp.has_surface_material ||
                    any(sp.emission > 0.f) ||
                    bounce + uint16_t(2) >= merian_render_max_path_length) {
                    break;
                }
                let follow_bsdf = sp.surface_material.get_bsdf();
                if (!bsdf_is_delta_only(follow_bsdf.get_lobe_types()) &&
                    follow_bsdf.effective_alpha >= merian_render_specular_alpha) {
                    break;
                }

                const OrthonormalFrame chain_frame = sp.shading_frame;
                const float3 chain_wi = chain_frame.to_local(-ray.Direction);
                const Optional<ShadingFunctionSample> chain_sample =
                    follow_bsdf.sample_eval(chain_wi, rng);
                if (!chain_sample.hasValue) {
                    break;
                }
                const float3 chain_wo_local = chain_sample.value.wo;
                const float3 chain_wo = chain_frame.to_world(chain_wo_local);
                const bool chain_transmission = chain_wi.z * chain_wo_local.z < 0.f;
                const Optional<RayDesc> chain_ray =
                    sp.make_ray(-ray.Direction, chain_wo, chain_transmission);
                if (!chain_ray.hasValue) {
                    break;
                }

                rec.vertex(params.path_records, sp.pos, chain_wo, chain_sample.value.pdf,
                           throughput * chain_bsdf * chain_sample.value.weight, uint(bounce),
                           path_record_pack_lobe(chain_sample.value.lobe) |
                               (chain_transmission ? PATH_RECORD_LOBE_TRANSMISSION : 0u),
                           PATH_RECORD_METHOD_BSDF, path_record_pack_material(sp.material_id));

                chain_bsdf *= chain_sample.value.weight;
                if (!chain_transmission) {
                    const float radius =
                        scene.get_curvature_radius(hit, scene.get_triangle_hit(hit));
                    mirror_focal_length = 0.5f * (chain_wi.z < 0.f ? -radius : radius);
                } else {
                    // apparent depth of a refraction (Ruppert et al. 2020); eta is incident over
                    // transmitted
                    let interior = sp.surface_material.get_interior_volume();
                    apparent_depth_scale *= interior.eta * abs(chain_wi.z / chain_wo_local.z);
                    if (merian_material_system_any_interior_volume) {
                        medium.cross_boundary(interior,
                                              dot(-ray.Direction, sp.get_face_normal()) >= 0.f);
                    }
                }
                ray = chain_ray.value;
                through_specular = true;
                bounce++;
            }

            segment.traced = true;
            segment.next_pos = sp.pos;
            segment.aim_pos =
                through_specular ? segment.pos + wo_world * apparent_distance : sp.pos;
            segment.next_normal = sp.shading_normal;
            segment.next_mv = half3((sp.pos - sp.prev_pos) / scene.time_diff);
            segment.incident = chain_bsdf * chain_transmittance * sp.emission;
            const float direct_mis = through_specular ? 1.f : mis_weight;
            segment.direct_scale = guiding_direct_scale(direct_mis);
            segment.weight = segment.pdf > 0.f ? f_cos / segment.pdf : float3(0);

            throughput *= segment.weight;
            const float3 emitted = throughput * segment.incident * direct_mis;
            radiance += emitted;
            rec.emission(emitted);
            throughput *= chain_bsdf;
        }

        if (merian_path_record && record_waits) {
            RayDesc probe;
            probe.Origin = offset_ray_origin(vertex.pos, ng, spawn_offset, wo_world);
            probe.Direction = wo_world;
            probe.TMin = SCENE_RAY_TMIN;
            probe.TMax = SCENE_RAY_TMAX;
            const Hit hit = scene.trace_ray(probe, RAY_FLAG_NONE, merian_render_instance_mask);
            const float light_hit_pdf =
                scene.nee.pdf(lights, scene.get_light_index(hit), probe.Origin, wo_world);
            const float light_pdf = p_light * scene.nee.mixture_pdf(light_hit_pdf);
            const float mixed_pdf = segment.pdf + light_pdf;
            rec.vertex(params.path_records, vertex.pos, wo_world, record_pdf + light_pdf,
                       mixed_pdf > 0.f ? record_f_cos / mixed_pdf : float3(0), uint(bounce) - 1u,
                       record_lobe, record_method, record_material);
        }

        // 4. guiding update
        const float3 tail =
            throughput *
            params.guiding.record_segment(guiding_state, frame_info, guiding_record, segment, rng);
        radiance += tail;
        rec.emission(tail);

        if (!next_ray.hasValue || !any(segment.weight > 0.f)) {
            break;
        }
    }

    rec.end(params.path_records, sp.pos);
    return radiance;
}

void render_pixel<Reorder : IThreadReorder>(const uint2 pixel,
                                            const ParameterBlock<Scene> scene,
                                            const ParameterBlock<RenderBinding> params) {
    if (not_on_texture(pixel, params.irradiance))
        return;

    const uint2 dim = params.gbuffer.get_dimensions();
    const Hit primary_hit = params.gbuffer.get_hit(pixel);

    const RayDesc primary_ray = scene.generate_camera_ray(pixel, dim);
    const ShadingPoint primary_sp = scene.get_shading_point(primary_ray, primary_hit);

    // demodulate per sample so the second moment stays consistent with the stored radiance
    const float3 demod_albedo =
        merian_render_demodulate_albedo ? params.gbuffer.get_demodulation_albedo(pixel) : float3(1);

    float3 irradiance = float3(0);
    float second_moment = 0.f;
    float3 scatter_stats = float3(0);

    for (int s = 0; s < merian_render_spp; s++) {
        RandomGenerator rng =
            PCG(xxhash32(uint4(pixel, scene.frame, uint(s))) ^ merian_render_seed);

        var rec = PathRecorder::begin(
            params.path_records, pixel, uint(s), scene.frame,
            uint(merian_render_max_path_length) *
                    (merian_render_nee_mode == NEE_MODE_RESAMPLED ? 2u : 1u) +
                1u);
        const float3 contrib = trace_path<Reorder>(scene, params, pixel, primary_ray, primary_sp,
                                                   uint16_t(s), rng, scatter_stats, rec) /
                               demod_albedo;
        if (!any(isnan(contrib)) && !any(isinf(contrib))) {
            irradiance += contrib;
            const float l = yuv_luminance(contrib);
            second_moment += l * l;
        }
    }

    if (merian_render_spp > 0) {
        irradiance /= float(merian_render_spp);
        second_moment /= float(merian_render_spp);
    }

    if (merian_render_output_view == OUTPUT_VIEW_SCATTER_STATISTICS) {
        // guided, refused by the shading function, refused a ray by the geometry
        params.irradiance[pixel] = float4(scatter_stats, 1);
        return;
    }

    if (merian_render_output_view == OUTPUT_VIEW_GUIDING_DEBUG) {
        const RenderFrameInfo frame_info = {scene.camera.position, scene.prev_camera.position,
                                            scene.time, scene.time_diff, scene.frame};
        RandomGenerator debug_rng = PCG(xxhash32(uint4(pixel, scene.frame, 0xDEB0)));

        PathVertex vertex;
        vertex.pos = primary_sp.pos;
        vertex.prev_pos = primary_sp.prev_pos;
        vertex.shading_frame = primary_sp.shading_frame;
        vertex.alpha = 1.f;
        vertex.bounce = uint16_t(1);
        vertex.sample_index = uint16_t(0);
        vertex.pixel = pixel;
        vertex.in_volume = false;
        vertex.geo_normal = primary_sp.shading_frame.get_normal();
        vertex.transmissive = false;

        float4 view;
        if (params.guiding.debug_view(frame_info, vertex, debug_rng, view)) {
            params.irradiance[pixel] = view;
            return;
        }
    }

    params.irradiance[pixel] = float4(irradiance, second_moment);
}

[shader("compute")][numthreads(8, 8, 1)] void main_compute(uint3 thread_id : SV_DispatchThreadID,
                                                           ParameterBlock<Scene> scene,
                                                           ParameterBlock<RenderBinding> params) {
    render_pixel<NoThreadReorder>(thread_id.xy, scene, params);
}

    [shader("raygeneration")] void main(ParameterBlock<Scene> scene,
                                        ParameterBlock<RenderBinding> params) {
    render_pixel<SerThreadReorder>(DispatchRaysIndex().xy, scene, params);
}
