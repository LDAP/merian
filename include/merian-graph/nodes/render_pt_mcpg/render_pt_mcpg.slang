import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.colors.debug;
import merian_shaders.colors.colorspaces;
import merian_shaders.light_cache.hashed_irradiance_cache;
import mc;

using merian;

namespace merian {
extern static const bool merian_render_emission_on_primary = true;
extern static const int merian_render_spp = 1;
extern static const int merian_render_max_path_length = 8;
extern static const uint merian_render_instance_mask = 0xFFu;
}

extern static const int debug_output_selector = 0;

extern static const uint lc_buffer_size = 4000000;
extern static const uint lc_probe_count = 4;
extern static const bool lc_stochastic_interpolation = false;

extern static const uint mc_adaptive_buffer_size = 32777259;
extern static const int mc_samples = 5;
extern static const float p_guiding = 0.9;
extern static const bool missing_light_heuristic = true;

struct RenderBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
    RWTexture2D<float4> debug;
    HashedIrradianceCache<lc_buffer_size, lc_probe_count, lc_stochastic_interpolation> irr_cache;
    MCPG<mc_adaptive_buffer_size> mcpg;
};

float3 trace_path(const ParameterBlock<Scene> scene,
                  const ParameterBlock<RenderBinding> params,
                  RayDesc ray,
                  ShadingPoint sp,
                  inout RandomGenerator rng) {
    const float3 cam_pos = scene.camera.position;
    const float3 prev_cam_pos = scene.prev_camera.position;

    float3 radiance = merian_render_emission_on_primary ? sp.material.emission : float3(0);
    float3 throughput = float3(1);
    float current_ior = 1.0f;

    for (uint16_t bounce = 1; bounce < merian_render_max_path_length; bounce++) {
        if (all(throughput < 1e-5))
            break;

        const float3 cur_pos = sp.pos;
        const OrthonormalFrame frame = sp.material.shading_frame;
        const float3 normal = frame.get_normal();
        let m_bsdf = sp.material.get_bsdf();
        const float3 wi_local = frame.to_local(-ray.Direction);

        // Resample one guiding lobe from the Markov-chain mixture (using RIS)
        MCState selected;
        float score_sum = 0.0;
        float scores[mc_samples];
        Distribution<float3> vmfs[mc_samples];
        for (int i = 0; i < mc_samples; i++) {
            const MCState s = params.mcpg.load(cur_pos, normal, cam_pos, scene.time, rng);
            score_sum += s.sum_w;
            // comparison with NaN is false (when s.sum_w == 0 and score_sum == 0)
            if (rng.next_float() < s.sum_w / score_sum) {
                selected = s;
                vmfs[i] = vmfs[0];
                scores[i] = scores[0];
                scores[0] = s.sum_w;
                vmfs[0] = s.get_vmf(cur_pos);
            } else {
                scores[i] = s.sum_w;
                vmfs[i] = s.get_vmf(cur_pos);
            }
        }

        // Sample wo from the BSDF or the selected lobe
        const bool guide = score_sum > 0.0 && rng.next_float() < p_guiding;
        float3 wo_world;
        float3 wo_local;
        if (guide) {
            wo_world = vmfs[0].sample(rng);
            wo_local = frame.to_local(wo_world);
        } else if (let local = m_bsdf.sample(wi_local, rng)) {
            wo_local = local;
            wo_world = frame.to_world(local);
            selected = MCState();
        } else {
            break;
        }

        // Stochastic MIS
        float wo_p = 0.0;
        if (score_sum > 0) {
            for (int i = 0; i < mc_samples; i++) {
                wo_p += scores[i] * vmfs[i].pdf(wo_world);
            }
            wo_p /= score_sum;
        }
        const float bsdf_w = score_sum > 0.0 ? (1.0 - p_guiding) : 1.0;
        wo_p = bsdf_w * m_bsdf.pdf(wi_local, wo_local) + p_guiding * wo_p;

        const float3 bsdf = m_bsdf.eval(wi_local, wo_local, rng);
        const bool valid = any(bsdf > 0.0);

        float3 incident = float3(0);
        float3 lc_incident = float3(0);
        float3 next_pos = cur_pos;

        if (valid) {
            const float3 ng = sp.get_face_normal();
            if ((dot(wo_world, ng) < 0.f) != (dot(-ray.Direction, ng) < 0.f))
                current_ior /= sp.material.get_eta();
            ray.Origin = cur_pos + (dot(wo_world, ng) < 0.f ? -ng : ng) * SCENE_RAY_OFFSET;
            ray.Direction = wo_world;
            ray.TMin = SCENE_RAY_TMIN;
            ray.TMax = SCENE_RAY_TMAX;
            sp = scene.trace_and_get_shading_point(ray, merian_render_instance_mask,
                                                   RenderContext(bounce, current_ior));
            next_pos = sp.pos;

            incident = sp.material.emission;

            lc_incident =
                any(incident > 0.0) || merian_render_max_path_length == 2
                    ? incident
                    : float3(params.irr_cache.get(next_pos, sp.material.shading_frame.get_normal(),
                                                  cam_pos, prev_cam_pos, rng));

            throughput *= bsdf / wo_p;
            radiance += throughput * incident;
        }

        const float mc_f = yuv_luminance(lc_incident * bsdf / wo_p);
        if (!isinf(mc_f) && !isnan(mc_f)) {
            params.irr_cache.update(cur_pos, normal,
                                    lc_incident * bsdf / wo_p /* / max(wo_p, 10.0)*/, cam_pos,
                                    prev_cam_pos, scene.frame, rng);

            if (rng.next_float() * score_sum < mc_f * mc_samples) {
                selected.add_sample(cur_pos, mc_f, next_pos, half3(0), scene.time);
                params.mcpg.store(selected, cur_pos, normal, cam_pos, scene.frame, rng);
            } else if (missing_light_heuristic && guide &&
                       selected.light_missing_heuristic(mc_f, wo_world, cur_pos, next_pos)) {

                params.mcpg.store(MCState(), cur_pos, normal, cam_pos, scene.frame, rng);
            }
        }

        if (!valid)
            break;
    }

    return radiance;
}

[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<RenderBinding> params) {
    const uint2 pixel = DispatchRaysIndex().xy;
    if (not_on_texture(pixel, params.irradiance))
        return;

    RandomGenerator rng = PCG(pcg3d16(uint3(pixel, scene.frame)));

    const RayDesc primary_ray = scene.generate_camera_ray(pixel, params.gbuffer.get_dimensions());
    const ShadingPoint primary_sp =
        scene.get_shading_point(primary_ray, params.gbuffer.get_hit(pixel));

    float3 irradiance = float3(0);
    float second_moment = 0.f;

    for (int s = 0; s < merian_render_spp; s++) {
        const float3 contrib = trace_path(scene, params, primary_ray, primary_sp, rng);

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

    const float3 cam_pos = scene.camera.position;
    const float3 prev_cam_pos = scene.prev_camera.position;
    switch (debug_output_selector) {
    case 1: { // moments
        params.debug[pixel] = float4(yuv_luminance(irradiance), second_moment.x, 0, 1);
        break;
    }
    case 2: { // light cache
        params.debug[pixel] = float4(
            params.irr_cache.get(primary_sp.pos, primary_sp.material.shading_frame.get_normal(),
                                 cam_pos, prev_cam_pos, rng),
            1);
        break;
    }
    case 3: { // mc grid
        const uint level = params.mcpg.grid.distributed_level_for_pos(primary_sp.pos, cam_pos, rng);
        const int3 gidx = params.mcpg.grid.grid_idx(
            primary_sp.pos, params.mcpg.grid.grid_width_for_level(level), rng);
        params.debug[pixel] = float4(random_color(params.mcpg.grid.buffer_index(gidx, level)), 1);
        break;
    }
    case 4: { // mc grid lod (level)
        params.debug[pixel] =
            params.mcpg.grid.distributed_level_for_pos(primary_sp.pos, cam_pos, rng) / 63.0;
        break;
    }
    case 5: { // mc weight
        const MCState mc_state =
            params.mcpg.load(primary_sp.pos, primary_sp.material.shading_frame.get_normal(),
                             cam_pos, scene.time, rng);
        params.debug[pixel] = float4(debug_color(mc_state.sum_w * 0.1), 1);
        break;
    }
    case 6: { // mc mean direction
        const MCState mc_state =
            params.mcpg.load(primary_sp.pos, primary_sp.material.shading_frame.get_normal(),
                             cam_pos, scene.time, rng);
        params.debug[pixel] = float4((mc_state.get_dir(primary_sp.pos) + 1) / 2, 1);
        break;
    }
    case 7: { // mc cos
        const MCState mc_state =
            params.mcpg.load(primary_sp.pos, primary_sp.material.shading_frame.get_normal(),
                             cam_pos, scene.time, rng);
        params.debug[pixel] = float4(
            mc_state.sum_w > 0 ? debug_color(mc_state.w_cos / mc_state.sum_w) : float3(0), 1);
        break;
    }
    case 8: { // mc N
        const MCState mc_state =
            params.mcpg.load(primary_sp.pos, primary_sp.material.shading_frame.get_normal(),
                             cam_pos, scene.time, rng);
        params.debug[pixel] = float4(float3(float(mc_state.N) / float(ML_MAX_N)), 1);
        break;
    }
    case 9: { // mc mv
        const MCState mc_state =
            params.mcpg.load(primary_sp.pos, primary_sp.material.shading_frame.get_normal(),
                             cam_pos, scene.time, rng);
        params.debug[pixel] = float4(float3(mc_state.mv), 1);
        break;
    }
    default: { // irradiance
        params.debug[pixel] = float4(irradiance, 1);
        break;
    }
    }
}
