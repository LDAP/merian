import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.colors.debug;
import merian_shaders.colors.colorspaces;
import mcpg_common;

using merian;

namespace merian {
extern static const bool merian_render_emission_on_primary = true;
extern static const int merian_render_spp = 1;
extern static const int merian_render_max_path_length = 8;
extern static const bool merian_render_demodulate_albedo = false;
} // namespace merian

extern static const int debug_output_selector = 0;

extern static const bool use_light_cache_tail = false;
extern static const float lc_min_pdf = 1.0;

extern static const float p_guiding = 0.9;

struct RenderBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
    RWTexture2D<float4> debug;
    RenderIrradianceCache irr_cache;
    RenderMCPG mcpg;
};

float3 trace_path(const ParameterBlock<Scene> scene,
                  const ParameterBlock<RenderBinding> params,
                  RayDesc ray,
                  ShadingPoint sp,
                  const int sample_i,
                  inout RandomGenerator rng) {
    const float3 cam_pos = scene.camera.position;
    const float3 prev_cam_pos = scene.prev_camera.position;

    // Medium the ray is travelling through; updated on transmission events.
    HomogeneousVolume medium = scene.get_exterior_medium();
    float current_ior = 1.0f;

    float3 throughput = medium.is_vacuum() ? float3(1)
                                           : medium.transmittance(distance(ray.Origin, sp.pos));
    float3 radiance = merian_render_emission_on_primary ? throughput * sp.emission : float3(0);

    // where this vertex was already looked up in the light cache, one bounce earlier
    HashGridPos lc_grid_pos = HashGridPos();

    for (uint16_t bounce = 1; bounce < merian_render_max_path_length; bounce++) {
        // the ray left the scene: nothing to scatter from, emission is already accumulated
        if (!sp.has_surface_material || all(throughput < SCENE_THROUGHPUT_THRESHOLD))
            break;

        const HashGridPos cur_lc_grid_pos = lc_grid_pos;
        lc_grid_pos = HashGridPos();

        const float3 cur_pos = sp.pos;
        // temporally reproject the cache lookup for the first sample so the lobe written last
        // frame is found despite geometry motion
        const float3 lookup_pos = select(sample_i == 0, sp.prev_pos, cur_pos);
        const OrthonormalFrame frame = sp.shading_frame;
        const float3 normal = frame.get_normal();
        let m_bsdf = sp.surface_material.get_bsdf();
        const float3 wi_local = frame.to_local(-ray.Direction);

        // Resample one guiding lobe from the Markov-chain mixture (using RIS)
        MCState selected;
        HashGridPos selected_pos;
        uint selected_level = 0;
        float score_sum = 0.0;
        float scores[mc_samples];
        VonMisesFisher3<false> vmfs[mc_samples];
        const float3 access_normal = params.mcpg.jitter_normal(frame, rng);
        for (int i = 0; i < mc_samples; i++) {
            HashGridPos grid_pos;
            uint level;
            const MCState s =
                params.mcpg.load(lookup_pos, access_normal, cam_pos, scene.time, rng, grid_pos, level);
            score_sum += s.sum_w;
            // comparison with NaN is false (when s.sum_w == 0 and score_sum == 0)
            if (rng.next_float() < s.sum_w / score_sum) {
                selected = s;
                selected_pos = grid_pos;
                selected_level = level;
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
            selected_pos = HashGridPos();
            selected_level = params.mcpg.grid.target_level_for_pos(cur_pos, cam_pos, rng);
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
        const bool transmission = wi_local.z * wo_local.z < 0.f;
        Optional<RayDesc> next_ray = none;
        if (any(bsdf > 0.0))
            next_ray = sp.make_ray(-ray.Direction, wo_world, transmission);

        float3 incident = float3(0);
        float3 lc_incident = float3(0);
        float3 next_pos = cur_pos;

        if (let next = next_ray) {
            if (transmission) {
                const HomogeneousVolume interior = sp.surface_material.get_interior_volume();
                current_ior /= interior.eta;
                // leaving a surface puts the ray back into the medium enclosing the scene
                medium = dot(-ray.Direction, sp.get_face_normal()) >= 0.f
                             ? interior
                             : scene.get_exterior_medium();
            }
            ray = next;
            // the last vertex is never scattered from, so its material sample would be dead
            sp = scene.trace_and_get_shading_point(
                ray, merian_render_instance_mask, RenderContext(bounce, current_ior),
                DefaultLODSampler(), bounce + 1 < merian_render_max_path_length);
            next_pos = sp.pos;

            incident = sp.emission;

            lc_incident =select(
                any(incident > 0.0) ||
                        (!use_light_cache_tail && merian_render_max_path_length == 2),
                     incident,
                     float3(params.irr_cache.get(next_pos, sp.shading_normal, cam_pos,
                                                 prev_cam_pos, rng, lc_grid_pos)));

            // Beer-Lambert attenuation over the segment just travelled; in-scattering is the
            // volume renderer's job. Folded into the incident radiance so that the guiding and the
            // irradiance cache see the attenuated signal.
            if (!medium.is_vacuum()) {
                const float3 segment_t = medium.transmittance(distance(ray.Origin, next_pos));
                incident *= segment_t;
                lc_incident *= segment_t;
            }

            throughput *= bsdf / wo_p;
            // complete the truncated path with the cached irradiance at the last vertex
            radiance += throughput *
                        select(use_light_cache_tail && bounce == merian_render_max_path_length - 1,
                               lc_incident, incident);
        }

        const float mc_f = yuv_luminance(lc_incident * bsdf / wo_p);
        if (!isinf(mc_f) && !isnan(mc_f)) {
            params.irr_cache.update(cur_pos, normal, lc_incident * bsdf / max(wo_p, lc_min_pdf),
                                    cam_pos, prev_cam_pos, scene.frame, rng, cur_lc_grid_pos);

            if (rng.next_float() * score_sum < mc_f * mc_samples) {
                const half3 target_mv = half3((sp.pos - sp.prev_pos) / scene.time_diff);
                selected.add_sample(cur_pos, mc_f, next_pos, target_mv, scene.time);
                params.mcpg.store(selected, cur_pos, access_normal, scene.camera.position,
                                  scene.frame, rng);

                // params.mcpg.store_at_level(selected, cur_pos, access_normal, selected_level + 1,
                //                            scene.frame, rng);
            } else if (missing_light_heuristic && selected_pos.is_valid() &&
                       selected.light_missing_heuristic(mc_f, wo_world, cur_pos, next_pos)) {

                params.mcpg.kill(selected_pos);
            }
        }

        if (!next_ray.hasValue)
            break;
    }

    return radiance;
}

void render_pixel(const uint2 pixel,
                  ParameterBlock<Scene> scene,
                  ParameterBlock<RenderBinding> params) {
    if (not_on_texture(pixel, params.irradiance))
        return;

    RandomGenerator rng = PCG(pcg3d16(uint3(pixel, scene.frame)));

    const RayDesc primary_ray = scene.generate_camera_ray(pixel, params.gbuffer.get_dimensions());
    const ShadingPoint primary_sp =
        scene.get_shading_point(primary_ray, params.gbuffer.get_hit(pixel));

    // demodulate per sample so the second moment stays consistent with the stored radiance
    const float3 demod_albedo = merian_render_demodulate_albedo
                                    ? params.gbuffer.get_demodulation_albedo(pixel)
                                    : float3(1);

    float3 irradiance = float3(0);
    float second_moment = 0.f;

    for (int sample_i = 0; sample_i < merian_render_spp; sample_i++) {
        const float3 contrib =
            trace_path(scene, params, primary_ray, primary_sp, sample_i, rng) / demod_albedo;

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
        if (let l = params.irr_cache.grid.get(primary_sp.pos, primary_sp.shading_normal, cam_pos,
                                              prev_cam_pos, rng)) {
            params.debug[pixel] = float4(l.irr, 1);
        } else {
            params.debug[pixel] = float4(1, 0, 1, 1);
        }

        break;
    }
    case 3: { // mc grid
        const uint level = params.mcpg.grid.distributed_level_for_pos(primary_sp.pos, cam_pos, rng);
        const int3 gidx = params.mcpg.grid.grid_idx(
            primary_sp.pos, params.mcpg.grid.grid_inv_width_for_level(level), rng);
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
            params.mcpg.load(primary_sp.pos, primary_sp.shading_frame, cam_pos, scene.time, rng);
        if (mc_state.is_valid())
            params.debug[pixel] = float4(debug_color(mc_state.sum_w * 0.1), 1);
        else
            params.debug[pixel] = float4(1, 0, 1, 1);
        break;
    }
    case 6: { // mc mean direction
        const MCState mc_state =
            params.mcpg.load(primary_sp.pos, primary_sp.shading_frame, cam_pos, scene.time, rng);
        if (mc_state.is_valid())
            params.debug[pixel] = float4((mc_state.get_dir(primary_sp.pos) + 1) / 2, 1);
        else
            params.debug[pixel] = float4(1, 0, 1, 1);
        break;
    }
    case 7: { // mc cos
        const MCState mc_state =
            params.mcpg.load(primary_sp.pos, primary_sp.shading_frame, cam_pos, scene.time, rng);
        if (mc_state.is_valid())
            params.debug[pixel] = float4(debug_color(mc_state.w_cos / mc_state.sum_w), 1);
        else
            params.debug[pixel] = float4(1, 0, 1, 1);
        break;
    }
    case 8: { // mc N
        const MCState mc_state =
            params.mcpg.load(primary_sp.pos, primary_sp.shading_frame, cam_pos, scene.time, rng);
        if (mc_state.is_valid())
            params.debug[pixel] = float4(float3(float(mc_state.N) / float(ML_MAX_N)), 1);
        else
            params.debug[pixel] = float4(1, 0, 1, 1);
        break;
    }
    case 9: { // mc mv
        const MCState mc_state =
            params.mcpg.load(primary_sp.pos, primary_sp.shading_frame, cam_pos, scene.time, rng);
        params.debug[pixel] = float4(float3(mc_state.mv), 1);
        break;
    }
    case 10: { // lc normal bin (actual)
        const float3 n = primary_sp.shading_normal;
        params.debug[pixel] = float4(random_color(params.irr_cache.grid.normal_bin(n)), 1);
        break;
    }
    case 11: { // lc normal bin (selected)
        const float3 n = primary_sp.shading_normal;
        const uint bin = params.irr_cache.grid.debug_selected_bin(primary_sp.pos, n, cam_pos, rng);
        params.debug[pixel] =
            bin == 0xFFFFFFFFu ? float4(0, 0, 0, 1) : float4(random_color(bin), 1);
        break;
    }
    default: { // irradiance
        params.debug[pixel] = float4(irradiance, 1);
        break;
    }
    }
}


// Two entry points over the same body: compute is faster (no RT-pipeline VGPR cap), while the
// raygen variant is available for experiments that need ray-tracing-pipeline-only features (SER).
[shader("compute")]
[numthreads(8, 8, 1)]
void main_compute(uint3 thread_id: SV_DispatchThreadID,
                  ParameterBlock<Scene> scene,
                  ParameterBlock<RenderBinding> params) {
    render_pixel(thread_id.xy, scene, params);
}

[shader("raygeneration")]
void main_rgen(ParameterBlock<Scene> scene, ParameterBlock<RenderBinding> params) {
    render_pixel(DispatchRaysIndex().xy, scene, params);
}
