import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.utils.volumes;
import merian_shaders.colors.colorspaces;
__exported import merian_shaders.sampling.guiding;

// Selects the guiding method at link time; render_pt.cpp injects the module.
__exported import render_pt_guiding;

using merian;

namespace merian {
extern static const uint merian_render_instance_mask = 0xFFu;
extern static const int merian_render_volume_spp = 1;
// light samples resampled into one shadow ray at the scattering vertex; 0 disables volume NEE
extern static const int merian_render_volume_nee_candidates = 1;
// Below this scattering distance the surface motion vector is the better estimate, so the forward
// projection leaves it alone.
extern static const float merian_render_volume_forward_project_min_z = 50.0;
} // namespace merian

struct VolumeBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> volume;
    RWTexture2D<float> volume_depth;
    RWTexture2D<float2> volume_mv;
    Sampler2D<float> prev_volume_depth;
    ParameterBlock<RenderGuiding> guiding;
    ParameterBlock<RenderDistanceGuiding> distance_guiding;
};

// Single scattering along the primary ray: the distance comes from transmittance sampling, the
// direction from the guiding mixed with the phase function, and the light from next event
// estimation mixed against whatever the scattered ray finds.
void render_pixel(const uint2 pixel,
                  const ParameterBlock<Scene> scene,
                  const ParameterBlock<VolumeBinding> params) {
    if (not_on_texture(pixel, params.volume))
        return;

    const uint2 dim = params.gbuffer.get_dimensions();
    let medium = scene.exterior_volume;
    let phase = medium.get_phase_function();
    const float mu_t = medium.sampling_mu_t();

    RandomGenerator rng = PCG(pcg3d16(uint3(pixel, scene.frame)));

    const RayDesc primary_ray = scene.generate_camera_ray(pixel, dim);
    const float linear_z = params.gbuffer.get_linear_z(pixel);
    // the surface pass owns everything behind the first hit
    const float max_t = min(linear_z, medium.get_max_distance());
    const float xi_max = transmittance_xi_max(max_t, mu_t);

    const RenderFrameInfo frame_info = {scene.camera.position, scene.prev_camera.position,
                                        scene.time, scene.time_diff, scene.frame};
    // a medium vertex has no surface, so the mixture is built around the view direction
    const OrthonormalFrame frame = OrthonormalFrame(-primary_ray.Direction);
    const float3 wi_local = frame.to_local(-primary_ray.Direction);

    float3 irradiance = float3(0);
    float second_moment = 0.f;
    // where along the ray the scattered light came from; it drives the reprojection
    float depth_sum = 0.f;
    float depth_weight = 0.f;

    for (uint16_t s = uint16_t(0); s < merian_render_volume_spp; s++) {
        // 1. a scattering distance out of the guided chains mixed with transmittance sampling
        const RayVertex ray_vertex = {primary_ray.Origin, primary_ray.Direction, max_t, pixel, s};
        var distance_state = params.distance_guiding.begin_path();
        let distances =
            params.distance_guiding.get_distribution(distance_state, frame_info, ray_vertex, rng);
        const float p_guided_distance =
            distances.is_valid() ? distances.get_selection_probability(1.f) : 0.f;

        const bool guided_distance = rng.next_float() < p_guided_distance;
        const float t = guided_distance ? distances.sample(rng)
                                        : transmittance_sample2(mu_t, rng.next_float(), xi_max);
        if (guided_distance && (t >= max_t || t <= 0.f)) {
            continue;
        }
        let distance_record = params.distance_guiding.record_scatter(distances, guided_distance);
        const float p_distance = (1.f - p_guided_distance) * transmittance_pdf2(t, mu_t, xi_max) +
                                 p_guided_distance * distances.pdf(t);

        const float3 scatter_pos = primary_ray.Origin + t * primary_ray.Direction;
        const float3 throughput = medium.transmittance(t) * medium.get_mu_s() / p_distance;

        // 2. the guiding lobes around this vertex, mixed with the phase function
        var guiding_state = params.guiding.begin_path();
        const PathVertex vertex = {scatter_pos, scatter_pos, frame, 1.f,
                                   uint16_t(1), s,           pixel, true};
        let guiding = params.guiding.get_distribution(guiding_state, frame_info, vertex, rng);
        const ScatterMixture mixture = ScatterMixture(guiding, 1.f);

        // 3. resampled direct light: the candidates share one shadow ray and die here
        DirectLightReservoir reservoir = DirectLightReservoir();
        const uint light_bucket = scene.get_light_bucket(pixel, scatter_pos);
        for (int i = 0; i < merian_render_volume_nee_candidates; i++) {
            if (let ls = scene.sample_light(light_bucket, scatter_pos, rng)) {
                const float3 wo_l = frame.to_local(ls.wo);
                reservoir.add_light(
                    ls, phase.eval(wi_local, wo_l, rng) * ls.Li * medium.transmittance(ls.dist),
                    merian_render_volume_nee_candidates * ls.pdf +
                        mixture.pdf(phase, guiding, wi_local, wo_l, ls.wo),
                    rng);
            }
        }

        // 4. one scattered direction out of the mixture
        bool scattered = false;
        bool guided = false;
        float3 wo_world = float3(0, 0, 1);
        float3 f_cos = float3(0);
        float scatter_pdf = 0.f;
        if (let sc = mixture.sample(phase, guiding, frame, wi_local, rng)) {
            wo_world = sc.wo;
            f_cos = sc.f_cos;
            scatter_pdf = sc.pdf;
            guided = sc.guided;
            scattered = true;
        }

        PathSegment segment;
        segment.pos = scatter_pos;
        segment.normal = frame.get_normal();
        segment.bounce = uint16_t(1);
        segment.wo = wo_world;
        segment.pdf = scatter_pdf;
        segment.weight = float3(0);
        segment.delta = false;
        segment.is_last = true;
        segment.traced = false;
        segment.next_pos = scatter_pos;
        segment.next_normal = segment.normal;
        segment.next_mv = half3(0);
        segment.incident = float3(0);

        // the record is what crosses the traversal, so it is taken while the candidates are dead
        let guiding_record = params.guiding.record_scatter(guiding, guided);

        float3 contribution = scene.resolve_direct_light(scatter_pos, float3(0), reservoir,
                                                         merian_render_instance_mask);

        if (scattered && any(f_cos > 0.f)) {
            RayDesc ray;
            ray.Origin = scatter_pos;
            ray.Direction = segment.wo;
            ray.TMin = SCENE_RAY_TMIN;
            ray.TMax = SCENE_RAY_TMAX;
            const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, merian_render_instance_mask);

            // the light sampler above could have generated this vertex too
            const LightHit lh = scene.get_light_hit(light_bucket, ray.Origin, ray.Direction, hit);
            const float mis_weight =
                segment.pdf / (merian_render_volume_nee_candidates * lh.pdf + segment.pdf);

            // single scattering never scatters again, so the material sample would be dead
            const ShadingPoint sp = scene.get_shading_point(ray, hit, RenderContext(uint16_t(1)),
                                                            DefaultLODSampler(), false);

            segment.traced = true;
            segment.next_pos = sp.pos;
            segment.next_normal = sp.shading_normal;
            segment.next_mv = half3((sp.pos - sp.prev_pos) / scene.time_diff);
            segment.incident = sp.emission * medium.transmittance(distance(ray.Origin, sp.pos));
            segment.weight = segment.pdf > 0.f ? f_cos / segment.pdf : float3(0);

            contribution += segment.weight * segment.incident * mis_weight;
        }

        // what the guiding learns, and the estimate it completes the path with
        contribution +=
            segment.weight *
            params.guiding.record_segment(guiding_state, frame_info, guiding_record, segment, rng);

        const float3 contrib = throughput * contribution;
        if (!any(isnan(contrib)) && !any(isinf(contrib))) {
            irradiance += contrib;
            const float l = yuv_luminance(contrib);
            second_moment += l * l;
            depth_sum += t * l;
            depth_weight += l;

            const RaySegment ray_segment = {t, l};
            params.distance_guiding.record_segment(distance_state, frame_info, distance_record,
                                                   ray_segment, rng);
        }
    }

    if (merian_render_volume_spp > 0) {
        irradiance /= float(merian_render_volume_spp);
        second_moment /= float(merian_render_volume_spp);
    }

    params.volume[pixel] = float4(irradiance, second_moment);
    params.volume_depth[pixel] = depth_weight > 0.f ? depth_sum / depth_weight : linear_z;
}

[shader("compute")][numthreads(8, 8, 1)] void
single_scattering(uint3 thread_id : SV_DispatchThreadID,
                  ParameterBlock<Scene> scene,
                  ParameterBlock<VolumeBinding> params) {
    render_pixel(thread_id.xy, scene, params);
}

    // Seeds the volume motion vectors with the surface ones. Separate from the forward projection,
    // which scatters its writes and would otherwise race with the seeding.
    [shader("compute")][numthreads(8, 8, 1)] void volume_project_seed(
        uint3 thread_id : SV_DispatchThreadID, ParameterBlock<VolumeBinding> params) {
    const uint2 pixel = thread_id.xy;
    if (any(pixel >= params.gbuffer.get_dimensions()))
        return;

    params.volume_mv[pixel] = params.gbuffer.get_motion_vectors(pixel);
}

// Surface motion vectors describe the first opaque hit, which is not where the volume signal comes
// from. Reproject last frame's mean scattering distance instead and overwrite the vectors it lands
// on. Concurrent writes to the same target pixel are benign: either is a valid estimate.
[shader("compute")][numthreads(8, 8, 1)] void volume_project(uint3 thread_id : SV_DispatchThreadID,
                                                             ParameterBlock<Scene> scene,
                                                             ParameterBlock<VolumeBinding> params) {
    const uint2 pixel = thread_id.xy;
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    const float prev_depth = params.prev_volume_depth.Load(int3(pixel, 0));
    if (prev_depth < merian_render_volume_forward_project_min_z)
        return;

    const float3 prev_dir = scene.prev_camera.get_pinhole_direction(pixel, dim);
    const float3 prev_pos = scene.prev_camera.position + prev_dir * prev_depth;
    const int2 new_pixel = int2(round(scene.camera.project_pinhole(prev_pos, dim)));
    if (any(new_pixel < 0) || any(new_pixel >= int2(dim)))
        return;

    params.volume_mv[uint2(new_pixel)] = float2(int2(pixel) - new_pixel);
}
