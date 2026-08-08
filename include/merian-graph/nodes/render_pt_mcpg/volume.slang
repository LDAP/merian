import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.utils.sampling;
import merian_shaders.utils.textures;
import merian_shaders.utils.volumes;
import merian_shaders.colors.colorspaces;
import merian_shaders.colors.debug;
import mcpg_common;
import mc_distance;

using merian;

extern static const int volume_spp = 1;
extern static const bool volume_use_light_cache = true;
extern static const int distance_mc_samples = 3;
// probability to sample the guiding distributions instead of transmittance / the phase function
extern static const float volume_p_dist_guiding = 0.0;
extern static const float volume_p_guiding = 0.7;

// Below this scattering distance the surface motion vector is the better estimate, so the forward
// projection leaves it alone.
extern static const float volume_forward_project_min_z = 50.0;

struct VolumeBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> volume;
    RWTexture2D<float> volume_depth;
    RWTexture2D<float2> volume_mv;
    RWTexture2D<float4> debug;
    Sampler2D<float> prev_volume_depth;
    RWStructuredBuffer<DistanceMCVertex> distance_mc;
    RenderIrradianceCache irr_cache;
    RenderMCPG mcpg;
};

// Single scattering along the primary ray. The distance comes from a per-pixel Markov chain mixed
// with transmittance sampling, the direction from the world-space chains the surface pass builds
// mixed with the phase function.
void render_pixel(const uint2 pixel,
                  ParameterBlock<Scene> scene,
                  ParameterBlock<VolumeBinding> params) {
    if (not_on_texture(pixel, params.volume))
        return;

    const uint2 dim = params.gbuffer.get_dimensions();
    let medium = scene.exterior_volume;
    const float mu_t = medium.sampling_mu_t();
    let phase = medium.get_phase_function();

    RandomGenerator rng = PCG(pcg3d16(uint3(pixel, scene.frame)));

    const float3 cam_pos = scene.camera.position;
    const float3 prev_cam_pos = scene.prev_camera.position;
    const RayDesc primary_ray = scene.generate_camera_ray(pixel, dim);
    const float linear_z = params.gbuffer.get_linear_z(pixel);
    const float2 mv = params.volume_mv[pixel];

    const uint grid_max_x = dim.x / distance_mc_grid_width + 1;
    const float max_t = min(linear_z, medium.get_max_distance());
    const float xi_max = transmittance_xi_max(max_t, mu_t);

    float3 irradiance = float3(0);
    float second_moment = 0.f;
    DistanceMCState dist_mc = distance_mc_state_new();

    for (int sample_i = 0; sample_i < volume_spp; sample_i++) {
        // 1. resample a distance chain (RIS) and sample a scattering distance from it
        dist_mc = distance_mc_state_new();
        float dist_scores[distance_mc_samples];
        float2 normal_dists[distance_mc_samples];
        float dist_score_sum = 0.0;
        float t = 0.0;
        for (int i = 0; i < distance_mc_samples; i++) {
            // the first sample follows the motion vector, the rest stay on this pixel
            const float2 lookup = select(sample_i == 0,
                                         clamp(float2(pixel) + mv, float2(0), float2(dim) - 1.f),
                                         float2(pixel));
            DistanceMCState s;
            distance_mc_load(params.distance_mc, s, lookup, grid_max_x, rng);
            normal_dists[i] = distance_mc_state_get_normal_dist(s);
            // the reprojected chain holds distances measured from the previous camera
            if (sample_i == 0)
                normal_dists[i].x -= dot(cam_pos - prev_cam_pos, primary_ray.Direction);

            dist_scores[i] = s.sum_w * float(s.sum_w > 0) * float(normal_dists[i].x < linear_z);
            dist_score_sum += dist_scores[i];
            // comparison with NaN is false (when dist_scores[i] == 0 and dist_score_sum == 0)
            if (rng.next_float() < dist_scores[i] / dist_score_sum) {
                dist_mc = s;
                t = sample_normal_box_muller(normal_dists[i].x, normal_dists[i].y, rng).x;
            }
        }

        const bool guide_dist = dist_score_sum > 0.0 && rng.next_float() < volume_p_dist_guiding;
        if (!guide_dist) {
            t = transmittance_sample2(mu_t, rng.next_float(), xi_max);
        } else if (t >= max_t || t <= 0.0) {
            continue;
        }

        // Stochastic MIS over the chain mixture and transmittance sampling
        float p = 0.0;
        if (dist_score_sum > 0.0) {
            for (int i = 0; i < distance_mc_samples; i++)
                p += dist_scores[i] * sample_normal_pdf(normal_dists[i].x, normal_dists[i].y, t);
            p /= dist_score_sum;
        }
        const float transmittance_w = dist_score_sum > 0.0 ? (1.0 - volume_p_dist_guiding) : 1.0;
        p = transmittance_w * transmittance_pdf2(t, mu_t, xi_max) + volume_p_dist_guiding * p;

        const float3 scatter_pos = primary_ray.Origin + t * primary_ray.Direction;

        // 2. resample one guiding lobe from the Markov-chain mixture (using RIS)
        MCState selected;
        HashGridPos selected_pos;
        uint selected_level = 0;
        float scores[mc_samples];
        VonMisesFisher3<false> vmfs[mc_samples];
        float score_sum = 0.0;
        // the chains are normal-binned; a volume point has no normal, so bin by the view direction
        const float3 access_normal = params.mcpg.jitter_normal(-primary_ray.Direction, rng);
        for (int i = 0; i < mc_samples; i++) {
            HashGridPos grid_pos;
            uint level;
            const MCState s = params.mcpg.load(scatter_pos, access_normal, cam_pos, scene.time, rng,
                                               grid_pos, level);
            score_sum += s.sum_w;
            if (rng.next_float() < s.sum_w / score_sum) {
                selected = s;
                selected_pos = grid_pos;
                selected_level = level;
                vmfs[i] = vmfs[0];
                scores[i] = scores[0];
                scores[0] = s.sum_w;
                vmfs[0] = s.get_vmf(scatter_pos);
            } else {
                scores[i] = s.sum_w;
                vmfs[i] = s.get_vmf(scatter_pos);
            }
        }

        // 3. sample wo from the phase function or the selected lobe
        const bool guide = score_sum > 0.0 && rng.next_float() < volume_p_guiding;
        float3 wo;
        if (guide) {
            wo = vmfs[0].sample(rng);
        } else {
            wo = phase.sample(-primary_ray.Direction, rng).value;
            selected = MCState();
            selected_pos = HashGridPos();
            selected_level = params.mcpg.grid.target_level_for_pos(scatter_pos, cam_pos, rng);
        }

        float wo_p = 0.0;
        if (score_sum > 0.0) {
            for (int i = 0; i < mc_samples; i++)
                wo_p += scores[i] * vmfs[i].pdf(wo);
            wo_p /= score_sum;
        }
        const float phase_value = phase.eval(dot(primary_ray.Direction, wo));
        const float phase_w = score_sum > 0.0 ? (1.0 - volume_p_guiding) : 1.0;
        wo_p = phase_w * phase_value + volume_p_guiding * wo_p;

        p *= wo_p;

        // 4. gather the incident radiance
        RayDesc ray;
        ray.Origin = scatter_pos;
        ray.Direction = wo;
        ray.TMin = SCENE_RAY_TMIN;
        ray.TMax = SCENE_RAY_TMAX;
        // single scattering never scatters off this surface, so the material sample would be dead
        const ShadingPoint sp = scene.trace_and_get_shading_point(
            ray, merian_render_instance_mask, RenderContext(uint16_t(1), 1.0f),
            DefaultLODSampler(), false);

        float3 incident = sp.emission;
        if (volume_use_light_cache && !any(incident > 0.0)) {
            HashGridPos lc_grid_pos;
            incident = float3(params.irr_cache.get(sp.pos, sp.shading_normal, cam_pos, prev_cam_pos,
                                                   rng, lc_grid_pos));
        }
        incident *= medium.transmittance(distance(scatter_pos, sp.pos));

        const float3 contrib = incident * phase_value * medium.get_mu_s() * medium.transmittance(t) / p;
        if (any(isnan(contrib)) || any(isinf(contrib)))
            continue;

        irradiance += contrib;
        const float l = yuv_luminance(contrib);
        second_moment += l * l;

        // 5. update the distance chain and the direction chain
        distance_mc_state_add_sample(dist_mc, t, l);
        if (rng.next_float() * dist_score_sum < l * distance_mc_samples)
            distance_mc_save(params.distance_mc, dist_mc, float2(pixel), grid_max_x, rng);

        const float mc_f = yuv_luminance(incident * phase_value / wo_p);
        if (isinf(mc_f) || isnan(mc_f))
            continue;

        if (rng.next_float() * score_sum < mc_f * mc_samples) {
            const half3 target_mv = half3((sp.pos - sp.prev_pos) / scene.time_diff);
            selected.add_sample(scatter_pos, mc_f, sp.pos, target_mv, scene.time);
            // sharp update at the queried level, plus a coarse propagation one level up
            params.mcpg.store(selected, scatter_pos, access_normal, cam_pos, scene.frame, rng);
            params.mcpg.store_at_level(selected, scatter_pos, access_normal, selected_level + 1,
                                       scene.frame, rng);
        } else if (missing_light_heuristic && selected_pos.is_valid() &&
                   selected.light_missing_heuristic(mc_f, wo, scatter_pos, sp.pos)) {
            params.mcpg.kill(selected_pos);
        }
    }

    if (volume_spp > 0) {
        irradiance /= float(volume_spp);
        second_moment /= float(volume_spp);
    }

    params.volume[pixel] = float4(irradiance, second_moment);

    // the stored chain at this pixel, at a fixed scale so the field stays comparable across
    // pixels (dividing by linear_z would imprint the geometry into the view)
    if (debug_output_selector == 12 || debug_output_selector == 13) {
        DistanceMCState s;
        distance_mc_load(params.distance_mc, s, float2(pixel), grid_max_x, rng);
        const float2 nd = distance_mc_state_get_normal_dist(s);
        const float v = select(debug_output_selector == 12, nd.x, nd.y);
        params.debug[pixel] = float4(debug_color(v / medium.get_max_distance()), 1);
    }
    // the mean scattering distance is where the volume signal lives; it drives the reprojection
    params.volume_depth[pixel] =
        dist_mc.sum_w > 0.0 ? dist_mc.moments.x / dist_mc.sum_w : linear_z;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void single_scattering(uint3 thread_id: SV_DispatchThreadID,
                 ParameterBlock<Scene> scene,
                 ParameterBlock<VolumeBinding> params) {
    render_pixel(thread_id.xy, scene, params);
}

// Seeds the volume motion vectors with the surface ones. Separate from the forward projection,
// which scatters its writes and would otherwise race with the seeding.
[shader("compute")]
[numthreads(8, 8, 1)]
void volume_project_seed(uint3 thread_id: SV_DispatchThreadID, ParameterBlock<VolumeBinding> params) {
    const uint2 pixel = thread_id.xy;
    if (any(pixel >= params.gbuffer.get_dimensions()))
        return;

    params.volume_mv[pixel] = params.gbuffer.get_motion_vectors(pixel);
}

// Surface motion vectors describe the first opaque hit, which is not where the volume signal comes
// from. Reproject last frame's mean scattering distance instead and overwrite the vectors it lands
// on. Concurrent writes to the same target pixel are benign: either is a valid estimate.
[shader("compute")]
[numthreads(8, 8, 1)]
void volume_project(uint3 thread_id: SV_DispatchThreadID,
                            ParameterBlock<Scene> scene,
                            ParameterBlock<VolumeBinding> params) {
    const uint2 pixel = thread_id.xy;
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    const float prev_depth = params.prev_volume_depth.Load(int3(pixel, 0));
    if (prev_depth < volume_forward_project_min_z)
        return;

    const float3 prev_dir = scene.prev_camera.get_pinhole_direction(pixel, dim);
    const float3 prev_pos = scene.prev_camera.position + prev_dir * prev_depth;
    const int2 new_pixel = int2(round(scene.camera.project_pinhole(prev_pos, dim)));
    if (any(new_pixel < 0) || any(new_pixel >= int2(dim)))
        return;

    params.volume_mv[uint2(new_pixel)] = float2(int2(pixel) - new_pixel);
}
