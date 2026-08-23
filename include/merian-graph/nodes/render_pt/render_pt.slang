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
extern static const int merian_render_spp = 1;
extern static const int merian_render_max_path_length = 8;
extern static const uint merian_render_instance_mask = 0xFFu;
extern static const bool merian_render_enable_ser = false;
extern static const bool merian_render_demodulate_albedo = false;
extern static const int merian_render_nee_mode = NEE_MODE_RESAMPLED;
// probability of replacing the scatter sample with a light sample (NEE_MODE_MIXTURE)
extern static const float merian_render_nee_probability = 0.5;
// light samples resampled into one shadow ray (NEE_MODE_RESAMPLED)
extern static const int merian_render_nee_candidates = 1;
// vertices (from the primary) that perform NEE; 0 = all
extern static const int merian_render_nee_bounces = 0;
} // namespace merian

struct RenderBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
    ParameterBlock<RenderGuiding> guiding;
}

float3
trace_path(const ParameterBlock<Scene> scene,
           const ParameterBlock<RenderBinding> params,
           const uint2 pixel,
           RayDesc ray,
           ShadingPoint sp,
           const uint16_t sample_index,
           inout RandomGenerator rng) {
    // Medium the ray is travelling through; updated on transmission events.
    var medium = scene.make_path_medium();

    float3 throughput = medium.transmittance(distance(ray.Origin, sp.pos));
    float3 radiance = merian_render_emission_on_primary ? throughput * sp.emission : float3(0);

    const RenderFrameInfo frame_info = {scene.camera.position, scene.prev_camera.position,
                                        scene.time, scene.time_diff, scene.frame};
    var guiding_state = params.guiding.begin_path();

    const bool nee = merian_render_nee_mode != NEE_MODE_OFF && scene.nee.has_lights();
    const uint candidates = uint(max(merian_render_nee_candidates, 1));

    for (uint16_t bounce = 1; bounce < merian_render_max_path_length; bounce++) {
        // the ray left the scene: nothing to scatter from, emission is already accumulated
        if (!sp.has_surface_material || all(throughput < SCENE_THROUGHPUT_THRESHOLD)) {
            break;
        }

        const bool nee_here =
            nee && (merian_render_nee_bounces <= 0 || bounce <= merian_render_nee_bounces);
        const uint light_bucket =
            nee_here ? scene.get_light_bucket(pixel, sp.pos) : LIGHT_INVALID_INDEX;
        // Drawn before the mixture below is built: the triangle record would otherwise be live
        // across the guiding candidates and the material sample at once.
        const bool light_technique = merian_render_nee_mode == NEE_MODE_MIXTURE && nee_here &&
                                     rng.next_float() < merian_render_nee_probability;
        uint sampled_light = LIGHT_INVALID_INDEX;
        float3 light_dir = float3(0, 0, 1);
        if (light_technique) {
            if (let lp = scene.sample_light_point(light_bucket, sp.pos, rng)) {
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
        const PathVertex vertex = {sp.pos, sp.prev_pos,  frame, bsdf.alpha,
                                   bounce, sample_index, pixel};

        // 1. the guiding lobes around this vertex, mixed with the BSDF
        let guiding = params.guiding.get_distribution(guiding_state, frame_info, vertex, rng);
        const ScatterMixture mixture = ScatterMixture(guiding, bsdf.alpha);

        // light sampling is wasted on a near-specular lobe: it is far more peaked than any light
        const bool nee_surface = nee_here && bsdf.alpha >= NEE_MIN_ALPHA;

        // 2. resampled direct light: M candidates share one shadow ray. Only the reservoir
        // survives to the trace below; the candidates die here.
        DirectLightReservoir reservoir = DirectLightReservoir();
        if (merian_render_nee_mode == NEE_MODE_RESAMPLED && nee_surface) {
            for (uint i = 0; i < candidates; i++) {
                if (let ls = scene.sample_light(light_bucket, sp.pos, rng)) {
                    const float3 wo_l = frame.to_local(ls.wo);
                    reservoir.add_light(
                        ls, bsdf.eval(wi_local, wo_l, rng) * ls.Li * medium.transmittance(ls.dist),
                        candidates * ls.pdf + mixture.pdf(bsdf, guiding, wi_local, wo_l, ls.wo),
                        rng);
                }
            }
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
            if (let s = mixture.sample(bsdf, guiding, frame, wi_local, rng)) {
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
        segment.next_normal = segment.normal;
        segment.next_mv = half3(0);
        segment.incident = float3(0);

        // Last thing before the ray: the record is what crosses the traversal, so it must not be
        // taken while the mixture densities above are still live.
        let guiding_record = params.guiding.record_scatter(guiding, guided);

        // The shadow ray goes out once the guiding candidates are dead: only the reservoir and
        // the narrowed record cross this traversal.
        radiance += throughput * scene.resolve_direct_light(sp.pos, sp.get_face_normal(), reservoir,
                                                            merian_render_instance_mask);

#if defined(MERIAN_SPIRV_CAP_SUPPORTED_ShaderInvocationReorderEXT)
        if (merian_render_enable_ser) {
            ReorderThread(0, 0);
        }
#endif

        // A transmission event crosses the surface; update the medium we are now in.
        const bool transmission = wi_local.z * wo_local.z < 0.f;
        Optional<RayDesc> next_ray = none;
        if (scattered && any(f_cos > 0.f)) {
            next_ray = sp.make_ray(-ray.Direction, wo_world, transmission);
        }

        if (let next = next_ray) {
            if (merian_material_system_any_interior_volume && transmission) {
                // leaving a surface puts the ray back into the medium enclosing the scene
                medium.cross_boundary(sp.surface_material.get_interior_volume(),
                                      dot(-ray.Direction, sp.get_face_normal()) >= 0.f);
            }
            ray = next;

            // the last vertex is never scattered from, so its material sample would be dead
            const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, merian_render_instance_mask);

            // resolved before the hit is shaded, so neither the hit nor the ray stay live across
            // it. A delta direction is one no light sampler could have produced.
            float mis_weight = 1.f;
            if (nee_here && !delta) {
                const LightHit lh =
                    scene.get_light_hit(light_bucket, ray.Origin, ray.Direction, hit);
                if (merian_render_nee_mode == NEE_MODE_RESAMPLED) {
                    // the candidates above sampled this vertex M times
                    mis_weight = segment.pdf / (candidates * lh.pdf + segment.pdf);
                } else {
                    segment.pdf += p_light * lh.pdf;
                    // a light sample that reached something else is occluded: the vertex the ray
                    // found was generated by neither technique
                    if (sampled_light != LIGHT_INVALID_INDEX && sampled_light != lh.index) {
                        segment.pdf = 0.f;
                    }
                }
            }

            sp = scene.get_shading_point(ray, hit, RenderContext(bounce), DefaultLODSampler(),
                                         !segment.is_last);

            // Beer-Lambert attenuation over the segment just travelled inside the medium.
            // In-scattering is not accounted for here; that is the volume renderer's job. Folded
            // into the incident radiance so that the guiding sees the attenuated signal.
            const float3 segment_transmittance = medium.transmittance(distance(ray.Origin, sp.pos));

            segment.traced = true;
            segment.next_pos = sp.pos;
            segment.next_normal = sp.shading_normal;
            segment.next_mv = half3((sp.pos - sp.prev_pos) / scene.time_diff);
            segment.incident = sp.emission * segment_transmittance;
            segment.weight = segment.pdf > 0.f ? f_cos / segment.pdf : float3(0);

            throughput *= segment.weight;
            // the guiding learns the plain incident radiance; only the image estimate weighs the
            // emission against the light samples
            radiance += throughput * segment.incident * mis_weight;
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

void render_pixel(const uint2 pixel,
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

    for (int s = 0; s < merian_render_spp; s++) {
        RandomGenerator rng = PCG(pcg4d16(uint4(pixel, scene.frame, s)));

        const float3 contrib =
            trace_path(scene, params, pixel, primary_ray, primary_sp, uint16_t(s), rng) /
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

    params.irradiance[pixel] = float4(irradiance, second_moment);
}

// Two entry points over the same body: compute is faster (no RT-pipeline VGPR cap), while the
// raygen variant is available for experiments that need ray-tracing-pipeline-only features (SER).
[shader("compute")][numthreads(8, 8, 1)] void main_compute(uint3 thread_id : SV_DispatchThreadID,
                                                           ParameterBlock<Scene> scene,
                                                           ParameterBlock<RenderBinding> params) {
    render_pixel(thread_id.xy, scene, params);
}

    [shader("raygeneration")] void main(ParameterBlock<Scene> scene,
                                        ParameterBlock<RenderBinding> params) {
    render_pixel(DispatchRaysIndex().xy, scene, params);
}
