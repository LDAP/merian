import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.utils.volumes;
import merian_shaders.colors.colorspaces;
__exported import "merian-graph/nodes/render_pt/render_pt_common.slang";
__exported import render_pt_guiding;

using merian;

namespace merian {
extern static const uint merian_render_seed = 0;
extern static const uint merian_render_instance_mask = 0xFFu;
extern static const int merian_render_volume_spp = 1;
extern static const NEEMode merian_render_volume_nee_mode = NEEMode::Resampled;
extern static const float merian_render_volume_nee_probability = 0.f;
extern static const float merian_render_volume_guiding_share = 0.5;
extern static const bool merian_render_volume_cache_tail = false;
extern static const float merian_render_distance_guiding_share = 0.5;
extern static const float merian_render_volume_forward_project_min_z = 50.0;

struct PhaseTarget<P : ShadingFunction, G : GuidingDistribution<float3>, V : HomogeneousVolume>
    : IDirectLightTarget {
    P phase;
    G guiding;
    ScatterMixture mixture;
    V medium;
    OrthonormalFrame frame;
    float3 wi_local;

    float3 unshadowed(const LightSample ls, inout RandomGenerator rng) {
        return phase.eval(wi_local, frame.to_local(ls.wo), rng) * ls.Li *
               medium.transmittance(ls.dist);
    }

    float scatter_pdf(const LightSample ls) {
        return mixture.pdf(phase, guiding, wi_local, frame.to_local(ls.wo), ls.wo);
    }
}

PhaseTarget<P, G, V>
make_phase_target<P : ShadingFunction, G : GuidingDistribution<float3>, V : HomogeneousVolume>(
    const P phase,
    const G guiding,
    const ScatterMixture mixture,
    const V medium,
    const OrthonormalFrame frame,
    const float3 wi_local) {
    PhaseTarget<P, G, V> target = {phase, guiding, mixture, medium, frame, wi_local};
    return target;
}
} // namespace merian

struct VolumeBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> volume;
    RWTexture2D<float> volume_depth;
    RWTexture2D<float2> volume_mv;
    RWTexture2D<float4> debug;
    Sampler2D<float> prev_volume_depth;
    ParameterBlock<RenderVolumeGuiding> guiding;
    ParameterBlock<RenderDistanceGuiding> distance_guiding;
    ParameterBlock<RenderIrradianceCache> irradiance_cache;
};

void render_pixel(const uint2 pixel,
                  const ParameterBlock<Scene> scene,
                  const ParameterBlock<VolumeBinding> params) {
    if (not_on_texture(pixel, params.volume))
        return;

    const uint2 dim = params.gbuffer.get_dimensions();
    let medium = scene.exterior_volume;
    let phase = medium.get_phase_function();
    const float mu_t = medium.sampling_mu_t();

    RandomGenerator rng = PCG(xxhash32(uint4(pixel, scene.frame, 0x564Fu)) ^ merian_render_seed);

    const RayDesc primary_ray = scene.generate_camera_ray(pixel, dim);
    const float linear_z = params.gbuffer.get_linear_z(pixel);
    const float max_t = min(linear_z, medium.get_max_distance());
    const float xi_max = transmittance_xi_max(max_t, mu_t);

    const RenderFrameInfo frame_info = {scene.camera.position, scene.prev_camera.position,
                                        scene.time, scene.time_diff, scene.frame};
    const OrthonormalFrame frame = OrthonormalFrame(-primary_ray.Direction);
    const float3 wi_local = frame.to_local(-primary_ray.Direction);

    const bool nee = merian_render_volume_nee_mode != NEEMode::Off && scene.nee.has_lights();
    const LightOrScatter techniques = LightOrScatter(
        merian_render_volume_nee_mode == NEEMode::Mixture && nee
            ? merian_render_volume_nee_probability
            : 0.f);

    float3 irradiance = float3(0);
    float second_moment = 0.f;
    float depth_sum = 0.f;
    float depth_weight = 0.f;

    for (uint16_t s = uint16_t(0); s < merian_render_volume_spp; s++) {
        // 1. scattering distance
        const RayVertex ray_vertex = {primary_ray.Origin, primary_ray.Direction, max_t, pixel, s};
        var distance_state = params.distance_guiding.begin_path();
        let distances =
            params.distance_guiding.get_distribution(distance_state, frame_info, ray_vertex, rng);
        const float p_guided_distance =
            distances.is_valid()
                ? clamp(merian_render_distance_guiding_share * distances.get_confidence(), 0.f,
                        1.f)
                : 0.f;

        const bool guided_distance = rng.next_float() < p_guided_distance;
        const float t = guided_distance ? distances.sample(rng)
                                        : transmittance_sample2(mu_t, rng.next_float(), xi_max);
        if (guided_distance && (t >= max_t || t <= 0.f)) {
            continue;
        }
        let distance_record =
            params.distance_guiding.record_scatter(distances, t, guided_distance, rng);
        const float p_distance = (1.f - p_guided_distance) * transmittance_pdf2(t, mu_t, xi_max) +
                                 p_guided_distance * distances.pdf(t);

        const float3 scatter_pos = primary_ray.Origin + t * primary_ray.Direction;
        const float3 throughput = medium.transmittance(t) * medium.get_mu_s() / p_distance;

        // 2. guiding
        var guiding_state = params.guiding.begin_path();
        const PathVertex vertex = {scatter_pos, scatter_pos, frame,       uint16_t(1),
                                   s,           pixel,       true,        frame.get_normal(),
                                   true};
        let guiding = params.guiding.get_distribution(guiding_state, frame_info, vertex, rng);
        const ScatterMixture mixture =
            ScatterMixture(guided_share(guiding, merian_render_volume_guiding_share, 1.f));

        // 3. resampled direct light
        let lights = scene.nee.begin_vertex(scatter_pos);
        Optional<DirectLightSample> direct = none;
        if (merian_render_volume_nee_mode == NEEMode::Resampled && nee) {
            direct = scene.sample_direct_light(
                lights, make_phase_target(phase, guiding, mixture, medium, frame, wi_local), rng);
        }

        // 4. scatter direction
        bool scattered = false;
        bool guided = false;
        uint sampled_light = LIGHT_INVALID_INDEX;
        uint sampled_entry = LIGHT_INVALID_INDEX;
        float3 wo_world = float3(0, 0, 1);
        float3 f_cos = float3(0);
        float scatter_pdf = 0.f;
        const bool light_scatter = techniques.draws_light(rng);
        if (light_scatter) {
            if (let lp = scene.nee.sample_point(lights, rng, sampled_entry)) {
                wo_world = normalize(lp.pos - scatter_pos);
                sampled_light = lp.index;
                const float3 wo_local = frame.to_local(wo_world);
                f_cos = phase.eval(wi_local, wo_local, rng);
                scatter_pdf = mixture.pdf(phase, guiding, wi_local, wo_local, wo_world);
                scattered = true;
            }
        } else if (let sc = sample_scatter(mixture, phase, guiding, frame, wi_local, rng)) {
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
        segment.pdf = techniques.scatter_density(scatter_pdf);
        segment.weight = float3(0);
        segment.delta = false;
        segment.is_last = !merian_render_volume_cache_tail;
        segment.traced = false;
        segment.chain = SPECULAR_CHAIN_EMPTY;
        segment.next_pos = scatter_pos;
        segment.aim_pos = scatter_pos;
        segment.next_normal = segment.normal;
        segment.next_mv = half3(0);
        segment.incident = float3(0);
        segment.direct_scale = guiding_direct_scale(1.f);
        let guiding_record = params.guiding.record_scatter(guiding, segment, guided, rng);

        float3 contribution = scene.resolve_direct_light(lights, float3(0), 0.f, direct,
                                                         merian_render_instance_mask);

        if (scattered && any(f_cos > 0.f)) {
            RayDesc ray;
            ray.Origin = scatter_pos;
            ray.Direction = segment.wo;
            ray.TMin = SCENE_RAY_TMIN;
            ray.TMax = SCENE_RAY_TMAX;
            const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, merian_render_instance_mask);

            float mis_weight = 1.f;
            float scatter_share = 1.f;
            bool light_reached = false;
            if (nee) {
                const uint light_index = scene.get_light_index(hit);
                const bool drawn = light_index == sampled_light;
                const float light_hit_pdf =
                    scene.nee.pdf(lights, light_index, ray.Origin, ray.Direction, hit.barycentrics,
                                  drawn, drawn ? sampled_entry : LIGHT_INVALID_INDEX);
                if (merian_render_volume_nee_mode == NEEMode::Resampled) {
                    mis_weight = balance_heuristic(segment.pdf, light_hit_pdf);
                    scatter_share = mis_weight;
                } else {
                    const float light_pdf =
                        techniques.light_density(scene.nee.mixture_pdf(light_hit_pdf));
                    scatter_share =
                        segment.pdf > 0.f ? balance_heuristic(segment.pdf, light_pdf) : 0.f;
                    segment.pdf += light_pdf;
                    light_reached = scene.nee.reaches(sampled_light, sampled_entry, light_index);
                    if (sampled_light != LIGHT_INVALID_INDEX && sampled_light != light_index) {
                        segment.pdf = 0.f;
                    }
                }
            }

            const ShadingPoint sp = scene.get_shading_point(ray, hit, RenderContext(uint16_t(1)),
                                                            DefaultLODSampler(), false);
            const float distance_travelled = distance(ray.Origin, sp.pos);
            if (sampled_light != LIGHT_INVALID_INDEX) {
                scene.nee.record_contribution(lights, sampled_light, wo_world,
                                              light_reached ? yuv_luminance(sp.emission) : 0.f,
                                              distance_travelled, sampled_entry);
            }

            segment.traced = true;
            segment.next_pos = sp.pos;
            segment.aim_pos = sp.pos;
            segment.next_normal = sp.shading_normal;
            segment.next_mv = half3((sp.pos - sp.prev_pos) / scene.time_diff);
            segment.incident = sp.emission;
            segment.direct_scale = guiding_direct_scale(scatter_share);
            const float3 transmittance = medium.transmittance(distance_travelled);
            segment.weight = segment.pdf > 0.f ? f_cos / segment.pdf * transmittance : float3(0);

            contribution += segment.weight * segment.incident * mis_weight;
            if (merian_render_volume_cache_tail && !any(sp.emission > 0.f)) {
                contribution += segment.weight * float3(params.irradiance_cache.get(
                                                     sp.pos, sp.shading_normal, frame_info.cam_pos,
                                                     frame_info.prev_cam_pos, rng));
            }
        } else if (light_scatter) {
            if (scattered)
                scene.nee.record_contribution(lights, sampled_light, wo_world, 0.f, 1.f,
                                              sampled_entry);
            else
                scene.nee.record_failed_draw(sampled_entry);
        }

        params.guiding.record_segment(guiding_state, frame_info, guiding_record, segment, rng);

        contribution *= throughput;
        if (!any(isnan(contribution)) && !any(isinf(contribution))) {
            irradiance += contribution;
            const float l = yuv_luminance(contribution);
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

    if (merian_render_debug_output == DebugOutput::DistanceGuiding) {
        RandomGenerator debug_rng = PCG(xxhash32(uint4(pixel, scene.frame, 0xDEB0)));
        const RayVertex ray_vertex = {primary_ray.Origin, primary_ray.Direction, max_t, pixel,
                                      uint16_t(0)};
        params.debug[pixel] =
            params.distance_guiding.debug_view(frame_info, ray_vertex, debug_rng);
    }
}

[shader("compute")][numthreads(8, 8, 1)] void
single_scattering(uint3 thread_id : SV_DispatchThreadID,
                  ParameterBlock<Scene> scene,
                  ParameterBlock<VolumeBinding> params) {
    render_pixel(thread_id.xy, scene, params);
}

// Separate from the forward projection, which scatters its writes.
[shader("compute")][numthreads(8, 8, 1)] void volume_project_seed(
        uint3 thread_id : SV_DispatchThreadID, ParameterBlock<VolumeBinding> params) {
    const uint2 pixel = thread_id.xy;
    if (any(pixel >= params.gbuffer.get_dimensions()))
        return;

    params.volume_mv[pixel] = params.gbuffer.get_motion_vectors(pixel);
}

// Concurrent writes to one pixel are benign: either is a valid estimate.
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
