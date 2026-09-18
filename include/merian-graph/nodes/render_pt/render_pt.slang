import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.colors.colorspaces;
__exported import merian_shaders.sampling.guiding;

// Selects the guiding method at link time; render_pt.cpp injects the module.
__exported import render_pt_guiding;

using merian;

namespace merian {
extern static const bool merian_render_emission_on_primary = true;
// which of the guiding method's own views to render instead of the image; 0 renders the image
extern static const int merian_render_guiding_debug_view = 0;
extern static const int merian_render_spp = 1;
// Decorrelates a run from another of the same configuration: the sample stream is otherwise a
// function of the pixel, the frame and the sample index alone.
extern static const uint merian_render_seed = 0;
extern static const int merian_render_max_path_length = 8;
extern static const uint merian_render_instance_mask = 0xFFu;
extern static const bool merian_render_enable_ser = false;
extern static const bool merian_render_demodulate_albedo = false;
// Follow a surface that only passes light on, instead of making it a path vertex: no lobe
// represents a delta BSDF and no light sampler can reach it.
extern static const bool merian_render_follow_specular = true;
// Also follow surfaces narrower than this; 0 follows only the singular ones.
extern static const float merian_render_specular_alpha = 0.f;
// Replace the image with per-pixel counts of the guided draws and the dead ones.
extern static const bool merian_render_scatter_stats = false;
extern static const int merian_render_nee_mode = NEE_MODE_RESAMPLED;
// probability of replacing the scatter sample with a light sample (NEE_MODE_MIXTURE)
extern static const float merian_render_nee_probability = 0.5;
// light samples resampled into one shadow ray (NEE_MODE_RESAMPLED)
// vertices (from the primary) that perform NEE; 0 = all
extern static const int merian_render_nee_bounces = 0;
// SCATTER_MODE_MIS draws the direction once from the mixture; SCATTER_MODE_RIS draws several and
// resamples them by how much the shading function makes of each
extern static const int merian_render_scatter_mode = SCATTER_MODE_MIS;
extern static const int merian_render_scatter_candidates = 2;
} // namespace merian

// Thread reordering is a ray tracing pipeline feature, so the call may only exist in the raygen
// specialization of the path loop.
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
}

// What a light candidate is worth here: its unshadowed contribution and the density the scatter
// techniques would have had for it.
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
        // a shadow ray towards a light the geometry hides would start behind the surface
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

float3 trace_path<Reorder : IThreadReorder>(const ParameterBlock<Scene> scene,
                                            const ParameterBlock<RenderBinding> params,
                                            const uint2 pixel,
                                            RayDesc ray,
                                            ShadingPoint sp,
                                            const uint16_t sample_index,
                                            inout RandomGenerator rng,
                                            inout float3 scatter_stats) {
    // Medium the ray is travelling through; updated on transmission events.
    var medium = scene.make_path_medium();

    float3 throughput = medium.transmittance(distance(ray.Origin, sp.pos));
    float3 radiance = merian_render_emission_on_primary ? throughput * sp.emission : float3(0);

    const RenderFrameInfo frame_info = {scene.camera.position, scene.prev_camera.position,
                                        scene.time, scene.time_diff, scene.frame};
    var guiding_state = params.guiding.begin_path();

    const bool nee = merian_render_nee_mode != NEE_MODE_OFF && scene.nee.has_lights();

    for (uint16_t bounce = 1; bounce < merian_render_max_path_length; bounce++) {
        // the ray left the scene: nothing to scatter from, emission is already accumulated
        if (!sp.has_surface_material || all(throughput < SCENE_THROUGHPUT_THRESHOLD)) {
            break;
        }

        const bool nee_here =
            nee && (merian_render_nee_bounces <= 0 || bounce <= merian_render_nee_bounces);
        let lights = scene.begin_direct_light(sp.pos, rng);
        // Drawn before the mixture below is built: the triangle record would otherwise be live
        // across the guiding candidates and the material sample at once.
        const bool light_technique = merian_render_nee_mode == NEE_MODE_MIXTURE && nee_here &&
                                     rng.next_float() < merian_render_nee_probability;
        uint sampled_light = LIGHT_INVALID_INDEX;
        float3 light_dir = float3(0, 0, 1);
        if (light_technique) {
            if (let lp = scene.sample_direct_light_point(lights, rng)) {
                // aimed from where the ray will start: a direction taken from the unoffset point
                // misses the sampled triangle near its edges
                const float3 ng = sp.get_face_normal();
                const float3 origin =
                    offset_ray_origin(sp.pos, dot(lp.pos - sp.pos, ng) < 0.f ? -ng : ng);
                light_dir = normalize(lp.pos - origin);
                sampled_light = lp.index;
            }
        }

        const OrthonormalFrame frame = sp.shading_frame;
        let bsdf = sp.surface_material.get_bsdf();
        const float3 wi_local = frame.to_local(-ray.Direction);
        const float3 ns = frame.get_normal();
        const float3 ng = sp.get_face_normal();
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

        // 1. the guiding lobes around this vertex, mixed with the BSDF
        let guiding = params.guiding.get_distribution(guiding_state, frame_info, vertex, rng);
        const ScatterMixture mixture =
            ScatterMixture(merian::guiding_share(guiding, bsdf.effective_alpha));

        // light sampling is wasted on a near-specular lobe: it is far more peaked than any light
        const bool nee_surface = nee_here && bsdf.effective_alpha >= NEE_MIN_ALPHA;

        // 2. resampled direct light: only the survivor reaches the shadow ray below
        Optional<DirectLightSample> direct = none;
        if (merian_render_nee_mode == NEE_MODE_RESAMPLED && nee_surface) {
            direct = scene.sample_direct_light(
                lights,
                make_scatter_target(bsdf, guiding, mixture, medium, frame, wi_local, ns, ng), rng);
        }

        // 3. one scatter direction: from a light, or out of the mixture. The light technique
        // replaces the scatter sample, so it is one ray either way; its density joins the mixture
        // once the hit says which light the ray reached.
        // The mass the light technique holds, whichever technique the draw above landed on: it is
        // the density that has to be charged, not the outcome.
        const float p_light = merian_render_nee_mode == NEE_MODE_MIXTURE && nee_surface
                                  ? merian_render_nee_probability
                                  : 0.f;
        // A vertex that generates no direction still owes its direct light, so it leaves through
        // the same path as one whose ray finds nothing: zero throughput, no continuation.
        bool scattered = false;
        bool guided = false;
        float3 wo_world = float3(0, 0, 1);
        float3 f_cos = float3(0);
        float scatter_pdf = 0.f;
        bool delta = false;
        if (light_technique && p_light > 0.f) {
            scattered = sampled_light != LIGHT_INVALID_INDEX;
            wo_world = light_dir;
            if (scattered) {
                const float3 wo_l = frame.to_local(wo_world);
                f_cos = bsdf.eval(wi_local, wo_l, rng);
                scatter_pdf = mixture.pdf(bsdf, guiding, wi_local, wo_l, wo_world);
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
                delta = s.delta;
                guided = s.guided;
                scattered = true;
            }
        }
        const float3 wo_local = frame.to_local(wo_world);

        // resolved before the continuation ray so that no candidate stays live across it
        PathSegment segment;
        segment.pos = vertex.pos;
        segment.normal = frame.get_normal();
        segment.bounce = bounce;
        segment.wo = wo_world;
        // a delta lobe already carries the only probability there is; the light term below is
        // zero for it either way
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
        segment.direct_scale = merian_guiding_direct_target == GUIDING_DIRECT_NONE ? 0.f : 1.f;

        // Last thing before the ray: the record is what crosses the traversal, so it must not be
        // taken while the mixture densities above are still live.
        let guiding_record = params.guiding.record_scatter(guiding, segment, guided, rng);

        // The shadow ray goes out once the guiding candidates are dead: only the reservoir and
        // the narrowed record cross this traversal.
        radiance += throughput * scene.resolve_direct_light(sp.pos, sp.get_face_normal(), direct,
                                                            merian_render_instance_mask);

        Reorder.reorder();

        // A transmission event crosses the surface; update the medium we are now in.
        const bool transmission = wi_local.z * wo_local.z < 0.f;
        Optional<RayDesc> next_ray = none;
        if (scattered && any(f_cos > 0.f)) {
            next_ray = sp.make_ray(-ray.Direction, wo_world, transmission);
        }

        if (merian_render_scatter_stats && guided) {
            scatter_stats += float3(1.f, select(all(f_cos <= 0.f), 1.f, 0.f),
                                    select(any(f_cos > 0.f) && !next_ray.hasValue, 1.f, 0.f));
        }

        if (let next = next_ray) {
            if (merian_material_system_any_interior_volume && transmission) {
                // leaving a surface puts the ray back into the medium enclosing the scene
                medium.cross_boundary(sp.surface_material.get_interior_volume(),
                                      dot(-ray.Direction, sp.get_face_normal()) >= 0.f);
            }
            ray = next;

            float mis_weight = 1.f;
            // what the surfaces followed through cost, and how far the light behind them is
            float3 chain_bsdf = float3(1);
            float3 chain_transmittance = float3(1);
            float unfolded = 0.f;
            // an interface compresses what is behind it, so every segment past one counts for less
            float unfold_scale = 1.f;
            // focal length of the mirror the last segment reflected off, 0 where it was flat
            float pending_focal = 0.f;
            bool through_specular = false;

            for (;;) {
                // the last vertex is never scattered from, so its material sample would be dead
                const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, merian_render_instance_mask);

                // Resolved before the hit is shaded, so neither stays live across it. Only the
                // first segment is weighed: no light sampler reaches past a singular surface.
                if (!through_specular && nee_surface && !delta) {
                    const LightHit lh = scene.get_light_hit(lights, ray.Origin, ray.Direction, hit);
                    if (merian_render_nee_mode == NEE_MODE_RESAMPLED) {
                        mis_weight = segment.pdf / (lh.pdf + segment.pdf);
                    } else {
                        segment.pdf += p_light * scene.direct_light_mixture_pdf(lh.pdf);
                        // a light sample that reached something else is occluded
                        if (sampled_light != LIGHT_INVALID_INDEX && sampled_light != lh.index) {
                            segment.pdf = 0.f;
                        }
                    }
                }

                sp = scene.get_shading_point(ray, hit, RenderContext(bounce), DefaultLODSampler(),
                                             !segment.is_last);

                // Beer-Lambert attenuation over the segment just travelled; no in-scattering.
                const float distance_travelled = distance(ray.Origin, sp.pos);
                chain_transmittance *= medium.transmittance(distance_travelled);
                // A curved mirror puts the image at its focal length R/2, not at the arc length.
                // One focusing in front of itself has no virtual image and keeps the distance.
                float segment_length = distance_travelled;
                if (pending_focal != 0.f) {
                    const float denominator = segment_length + pending_focal;
                    if (denominator > 1e-4f) {
                        segment_length *= pending_focal / denominator;
                    }
                    pending_focal = 0.f;
                }
                unfolded += unfold_scale * segment_length;

                if (!merian_render_follow_specular || !sp.has_surface_material ||
                    bounce + uint16_t(2) >= merian_render_max_path_length) {
                    break;
                }
                // a surface that emits is a source, not a window onto one
                if (any(sp.emission > 0.f)) {
                    break;
                }
                let follow_bsdf = sp.surface_material.get_bsdf();
                if (!bsdf_is_delta_only(follow_bsdf.get_lobe_types()) &&
                    follow_bsdf.effective_alpha >= merian_render_specular_alpha) {
                    break;
                }

                // follow it through: the BSDF alone, no guiding and no light sampling
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

                chain_bsdf *= chain_sample.value.weight;
                if (!chain_transmission) {
                    // the focal length of a mirror is half its radius, towards the side the ray is
                    const float radius =
                        scene.get_curvature_radius(hit, scene.get_triangle_hit(hit));
                    pending_focal = 0.5f * (chain_wi.z < 0.f ? -radius : radius);
                }
                // Apparent-depth factor of a refraction (Ruppert et al. 2020): it needs only the
                // relative index, so a dielectric enclosing no volume still gets it.
                if (chain_transmission) {
                    let interior = sp.surface_material.get_interior_volume();
                    // eta here is incident over transmitted, so it multiplies: entering
                    // glass the image is nearer, leaving it the image is farther
                    unfold_scale *= interior.eta * abs(chain_wi.z / chain_wo_local.z);
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
            // aimed at where the light is, not at the surface that passed it on
            segment.next_pos = sp.pos;
            segment.aim_pos = through_specular ? segment.pos + wo_world * unfolded : sp.pos;
            segment.next_normal = sp.shading_normal;
            segment.next_mv = half3((sp.pos - sp.prev_pos) / scene.time_diff);
            // what arrived along this direction, the surfaces followed through included
            segment.incident = chain_bsdf * chain_transmittance * sp.emission;
            const float direct_mis = through_specular ? 1.f : mis_weight;
            segment.direct_scale = merian_guiding_direct_target == GUIDING_DIRECT_MIS ? direct_mis
                                   : merian_guiding_direct_target == GUIDING_DIRECT_NONE ? 0.f
                                                                                         : 1.f;
            segment.weight = segment.pdf > 0.f ? f_cos / segment.pdf : float3(0);

            throughput *= segment.weight;
            // the guiding learns what the direct light policy allows, the image always weighs
            radiance += throughput * segment.incident * direct_mis;
            // the surfaces followed through are already in the radiance above
            throughput *= chain_bsdf;
        }

        // 4. what the guiding learns, and the estimate it completes the path with
        const float3 tail =
            params.guiding.record_segment(guiding_state, frame_info, guiding_record, segment, rng);
        radiance += throughput * tail;

        if (!next_ray.hasValue || !any(segment.weight > 0.f)) {
            break;
        }
    }

    return radiance;
}

void render_pixel<Reorder : IThreadReorder>(const uint2 pixel,
                                            const ParameterBlock<Scene> scene,
                                            const ParameterBlock<RenderBinding> params) {
    if (not_on_texture(pixel, params.irradiance))
        return;

    const uint2 dim = params.gbuffer.get_dimensions();
    const Hit primary_hit = params.gbuffer.get_hit(pixel);

    // primary ray and shading point are identical for every sample
    const RayDesc primary_ray = scene.generate_camera_ray(pixel, dim);
    const ShadingPoint primary_sp = scene.get_shading_point(primary_ray, primary_hit);

    // demodulate per sample so the second moment stays consistent with the stored radiance
    const float3 demod_albedo =
        merian_render_demodulate_albedo ? params.gbuffer.get_demodulation_albedo(pixel) : float3(1);

    float3 irradiance = float3(0);
    float second_moment = 0.f;
    float3 scatter_stats = float3(0);

    for (int s = 0; s < merian_render_spp; s++) {
        RandomGenerator rng = PCG(pcg4d16(uint4(pixel, scene.frame, s)) ^ merian_render_seed);

        const float3 contrib = trace_path<Reorder>(scene, params, pixel, primary_ray, primary_sp,
                                                   uint16_t(s), rng, scatter_stats) /
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

    if (merian_render_scatter_stats) {
        // guided, refused by the shading function, refused a ray by the geometry
        params.irradiance[pixel] = float4(scatter_stats, 1);
        return;
    }

    if (merian_render_guiding_debug_view > 0) {
        // the method answers for the primary hit
        const RenderFrameInfo frame_info = {scene.camera.position, scene.prev_camera.position,
                                            scene.time, scene.time_diff, scene.frame};
        RandomGenerator debug_rng = PCG(pcg4d16(uint4(pixel, scene.frame, 0xDEB0)));

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
        if (params.guiding.debug_view(merian_render_guiding_debug_view, frame_info, vertex,
                                      debug_rng, view)) {
            params.irradiance[pixel] = view;
            return;
        }
    }

    params.irradiance[pixel] = float4(irradiance, second_moment);
}

// Two entry points over the same body: compute is faster (no RT-pipeline VGPR cap), while the
// raygen variant is available for experiments that need ray-tracing-pipeline-only features (SER).
[shader("compute")][numthreads(8, 8, 1)] void main_compute(uint3 thread_id : SV_DispatchThreadID,
                                                           ParameterBlock<Scene> scene,
                                                           ParameterBlock<RenderBinding> params) {
    render_pixel<NoThreadReorder>(thread_id.xy, scene, params);
}

    [shader("raygeneration")] void main(ParameterBlock<Scene> scene,
                                        ParameterBlock<RenderBinding> params) {
    render_pixel<SerThreadReorder>(DispatchRaysIndex().xy, scene, params);
}
