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
extern static const int merian_render_nee_mode = NEE_MODE_MIXTURE;
// probability of replacing the scatter sample with a light sample (NEE_MODE_MIXTURE)
extern static const float merian_render_nee_probability = 0.5;
// light samples resampled into one shadow ray (NEE_MODE_RESAMPLED)
extern static const int merian_render_nee_candidates = 1;
// vertices (from the primary) that perform NEE; 0 = all
extern static const int merian_render_nee_bounces = 0;
} // namespace merian

extern static const bool use_light_cache_tail = false;
extern static const float lc_min_pdf = 1.0;
static const bool use_light_cache = merian_render_max_path_length > 2 || use_light_cache_tail;
static const bool recursive_light_cache =
    merian_render_max_path_length > 3 || use_light_cache_tail;

extern static const float p_guiding = 0.9;

static const int GUIDING_DIRECTIONAL_SAMPLING_MIS = 0;
static const int GUIDING_DIRECTIONAL_SAMPLING_MIS_ROUGHNESS = 1;
extern static const int guiding_directional_sampling_type =
    GUIDING_DIRECTIONAL_SAMPLING_MIS_ROUGHNESS;

// use guiding with GGX alpha >= this
extern static const float guiding_alpha_threshold = 0.05;

struct RenderBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
    RWTexture2D<float4> debug;
    RenderIrradianceCache irr_cache;
    RenderMCPG mcpg;
};

// Density of the guided/BSDF mixture that samples the scatter direction (stochastic MIS over the
// RIS candidates).
float directional_pdf<B : BSDF>(const B bsdf,
                                const float3 wi_local,
                                const float3 wo_local,
                                const float3 wo_world,
                                const float scores[mc_samples],
                                const VonMisesFisher3<false> vmfs[mc_samples],
                                const float score_sum,
                                const bool can_guide,
                                const float p_guide) {
    float guide_p = 0.0;
    if (can_guide) {
        for (int i = 0; i < mc_samples; i++) {
            guide_p += scores[i] * vmfs[i].pdf(wo_world);
        }
        guide_p /= score_sum;
    }
    const float bsdf_w = can_guide ? (1.0 - p_guide) : 1.0;
    return bsdf_w * bsdf.pdf(wi_local, wo_local) + select(can_guide, p_guide, 0.0) * guide_p;
}

float3 trace_path(const ParameterBlock<Scene> scene,
                  const ParameterBlock<RenderBinding> params,
                  RayDesc ray,
                  ShadingPoint sp,
                  const int sample_i,
                  inout RandomGenerator rng) {
    const float3 cam_pos = scene.camera.position;
    const float3 prev_cam_pos = scene.prev_camera.position;

    // Medium the ray is travelling through; updated on transmission events.
    HomogeneousVolume medium = scene.exterior_volume;

    float3 throughput = medium.transmittance(distance(ray.Origin, sp.pos));
    float3 radiance = merian_render_emission_on_primary ? throughput * sp.emission : float3(0);

    // where this vertex was already looked up in the light cache, one bounce earlier
    HashGridPos lc_grid_pos = HashGridPos();

    const bool nee = merian_render_nee_mode != NEE_MODE_OFF && scene.nee.has_lights();
    const uint nee_candidates = uint(max(merian_render_nee_candidates, 1));

    for (uint16_t bounce = 1; bounce < merian_render_max_path_length; bounce++) {
        // the ray left the scene: nothing to scatter from, emission is already accumulated
        if (!sp.has_surface_material || all(throughput < SCENE_THROUGHPUT_THRESHOLD))
            break;

        const HashGridPos cur_lc_grid_pos = lc_grid_pos;
        lc_grid_pos = HashGridPos();

        const float3 cur_pos = sp.pos;
        const float3 cur_face_normal = sp.get_face_normal();
        // temporally reproject the cache lookup for the first sample so the lobe written last
        // frame is found despite geometry motion
        const float3 lookup_pos = select(sample_i == 0, sp.prev_pos, cur_pos);
        const OrthonormalFrame frame = sp.shading_frame;
        const float3 normal = frame.get_normal();
        let m_bsdf = sp.surface_material.get_bsdf();
        const float3 wi_local = frame.to_local(-ray.Direction);

        const bool nee_here =
            nee && (merian_render_nee_bounces <= 0 || bounce <= merian_render_nee_bounces);
        // Sampled up front so the triangle record does not overlap the guiding candidates below.
        const float p_light = merian_render_nee_mode == NEE_MODE_MIXTURE && nee_here &&
                                      m_bsdf.alpha >= NEE_MIN_ALPHA
                                  ? merian_render_nee_probability
                                  : 0.f;
        const bool light_technique = p_light > 0.f && rng.next_float() < p_light;
        uint sampled_light = LIGHT_INVALID_INDEX;
        float3 light_dir = float3(0, 0, 1);
        if (light_technique) {
            if (let ls = scene.sample_light_point(cur_pos, rng)) {
                // aiming at the point keeps the ray on the light at grazing angles
                light_dir = normalize(ls.pos - cur_pos);
                sampled_light = ls.index;
            }
        }

        // Resample one guiding lobe from the Markov-chain mixture (using RIS)
        MCState selected;
        HashGridPos selected_pos;
        uint selected_level = 0;
        float score_sum = 0.0;
        float scores[mc_samples];
        VonMisesFisher3<false> vmfs[mc_samples];
        const float3 access_normal = frame.get_normal();
        for (int i = 0; i < mc_samples; i++) {
            HashGridPos grid_pos;
            uint level;
            const MCState s = params.mcpg.load(lookup_pos, access_normal, cam_pos, scene.time, rng,
                                               grid_pos, level);
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

        const float bsdf_alpha = m_bsdf.alpha;
        const bool can_guide = score_sum > 0.0 && bsdf_alpha >= guiding_alpha_threshold;
        const float p_guide =
            select(guiding_directional_sampling_type == GUIDING_DIRECTIONAL_SAMPLING_MIS_ROUGHNESS,
                   p_guiding * bsdf_alpha, p_guiding);

        // Resampled direct light: M candidates share one shadow ray. Resolved before the scatter
        // ray below, so none of it stays live across that traversal.
        if (merian_render_nee_mode == NEE_MODE_RESAMPLED && nee_here) {
            DirectLightReservoir reservoir = DirectLightReservoir();
            for (uint i = 0; i < nee_candidates; i++) {
                if (let ls = scene.sample_light(cur_pos, rng)) {
                    const float3 wo_l = frame.to_local(ls.wo);
                    reservoir.add_light(
                        ls,
                        m_bsdf.eval(wi_local, wo_l, rng) * ls.Li * medium.transmittance(ls.dist),
                        nee_candidates * ls.pdf + directional_pdf(m_bsdf, wi_local, wo_l, ls.wo,
                                                                  scores, vmfs, score_sum,
                                                                  can_guide, p_guide),
                        rng);
                }
            }
            radiance += throughput * scene.resolve_direct_light(cur_pos, cur_face_normal, reservoir,
                                                                merian_render_instance_mask);
        }

        // Scatter direction from the light sampled above, the selected lobe or the BSDF: one ray
        // either way. A failed sample ends the path.
        bool stop = false;
        bool delta = false;
        float3 delta_weight = float3(0);
        float3 wo_world = float3(0, 0, 1);
        float3 wo_local = float3(0, 0, 1);
        if (light_technique) {
            // a light sample that could not be drawn ends the path: the scatter technique did
            // not run, so no vertex was generated
            stop = sampled_light == LIGHT_INVALID_INDEX;
            wo_world = light_dir;
            wo_local = frame.to_local(wo_world);
            selected = MCState();
            selected_pos = HashGridPos();
            selected_level = params.mcpg.grid.target_level_for_pos(cur_pos, cam_pos, rng);
        } else if (can_guide && rng.next_float() < p_guide) {
            wo_world = vmfs[0].sample(rng);
            wo_local = frame.to_local(wo_world);
        } else if (let s = m_bsdf.sample_eval(wi_local, rng)) {
            wo_local = s.wo;
            wo_world = frame.to_world(s.wo);
            delta = s.lobe == BSDFLobe::Delta;
            delta_weight = s.weight;
            selected = MCState();
            selected_pos = HashGridPos();
            selected_level = params.mcpg.grid.target_level_for_pos(cur_pos, cam_pos, rng);
        } else {
            stop = true;
        }

        // A delta direction has no density under the guiding lobes or the lights, so all that
        // is left of the mixture is the chance of having drawn from the BSDF at all. Otherwise
        // the light density is only known once the hit is in, the rest of the mixture now.
        const float bsdf_w = can_guide ? (1.0 - p_guide) : 1.0;
        const float p_scatter =
            delta ? bsdf_w
                  : (1.f - p_light) * directional_pdf(m_bsdf, wi_local, wo_local, wo_world, scores,
                                                      vmfs, score_sum, can_guide, p_guide);

        const float3 bsdf = stop    ? float3(0)
                            : delta ? delta_weight
                                    : m_bsdf.eval(wi_local, wo_local, rng);
        const bool transmission = wi_local.z * wo_local.z < 0.f;
        Optional<RayDesc> next_ray = none;
        if (any(bsdf > 0.0))
            next_ray = sp.make_ray(-ray.Direction, wo_world, transmission);

        float3 incident = float3(0);
        float3 lc_incident = float3(0);
        float3 next_pos = cur_pos;
        float wo_p = p_scatter;

        if (let next = next_ray) {
            if (transmission) {
                const IsotropicVolume interior = sp.surface_material.get_interior_volume();
                // leaving a surface puts the ray back into the medium enclosing the scene
                if (dot(-ray.Direction, cur_face_normal) >= 0.f)
                    medium = interior;
                else
                    medium = scene.exterior_volume;
            }
            ray = next;
            // the last vertex is never scattered from, so its material sample would be dead
            const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, merian_render_instance_mask);

            // resolved before the hit is shaded, so neither the hit nor the ray stay live across
            // it
            float mis_weight = 1.f;
            if (nee_here && !delta) {
                const LightHit lh = scene.get_light_hit(ray.Origin, ray.Direction, hit);
                if (merian_render_nee_mode == NEE_MODE_MIXTURE) {
                    wo_p += p_light * lh.pdf;
                    // a light sample that reached something else is occluded: the vertex the ray
                    // found was generated by neither technique
                    stop =
                        stop || (sampled_light != LIGHT_INVALID_INDEX && sampled_light != lh.index);
                } else {
                    // the candidates above sampled this vertex M times
                    mis_weight = wo_p / (nee_candidates * lh.pdf + wo_p);
                }
            }

            sp = scene.get_shading_point(ray, hit, RenderContext(bounce),
                                         DefaultLODSampler(),
                                         bounce + 1 < merian_render_max_path_length);
            next_pos = sp.pos;

            incident = sp.emission;
            const bool emissive = any(incident > 0.0);

            lc_incident = select(
                emissive || (!use_light_cache_tail && merian_render_max_path_length == 2), incident,
                float3(params.irr_cache.get(next_pos, sp.shading_normal, cam_pos, prev_cam_pos, rng,
                                            lc_grid_pos)));

            // Beer-Lambert attenuation over the segment just travelled; in-scattering is the
            // volume renderer's job. Folded into the incident radiance so that the guiding and the
            // irradiance cache see the attenuated signal.
            const float3 segment_t = medium.transmittance(distance(ray.Origin, next_pos));
            incident *= segment_t;
            lc_incident *= segment_t;

            // the guiding and the irradiance cache learn the plain incident radiance; only the
            // image estimate weighs the emission against the light samples
            throughput *= bsdf / wo_p;
            const bool tail =
                use_light_cache_tail && bounce == merian_render_max_path_length - 1 && !emissive;
            if (!stop) {
                // complete the truncated path with the cached irradiance at the last vertex
                radiance += throughput * select(tail, lc_incident, incident * mis_weight);
            }
        }

        const float mc_f = yuv_luminance(lc_incident * bsdf / wo_p);
        if (!isinf(mc_f) && !isnan(mc_f)) {
            if (use_light_cache) {
                const float3 irr = select(recursive_light_cache, lc_incident, incident);
                params.irr_cache.update(cur_pos, normal, irr * bsdf / max(wo_p, lc_min_pdf),
                                        cam_pos, prev_cam_pos, scene.frame, rng, cur_lc_grid_pos);
            }

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

        if (stop || !next_ray.hasValue)
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
    const float3 demod_albedo =
        merian_render_demodulate_albedo ? params.gbuffer.get_demodulation_albedo(pixel) : float3(1);

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
        const uint level = params.irr_cache.grid.target_level_for_pos(primary_sp.pos, cam_pos, rng);
        params.debug[pixel] = float4(random_color(params.irr_cache.grid.normal_bin(n, level)), 1);
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
[shader("compute")][numthreads(8, 8, 1)] void main_compute(uint3 thread_id : SV_DispatchThreadID,
                                                           ParameterBlock<Scene> scene,
                                                           ParameterBlock<RenderBinding> params) {
    render_pixel(thread_id.xy, scene, params);
}

    [shader("raygeneration")] void main_rgen(ParameterBlock<Scene> scene,
                                             ParameterBlock<RenderBinding> params) {
    render_pixel(DispatchRaysIndex().xy, scene, params);
}
