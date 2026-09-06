#include "merian-graph/nodes/render_restir_di/render_restir_di.slangh"

using merian;

namespace merian {
extern static const bool merian_render_emission_on_primary = true;
extern static const bool merian_render_demodulate_albedo = false;
extern static const int merian_restir_spp = 4;
extern static const int merian_restir_spatial_iterations = 1;
extern static const bool merian_restir_apply_motion = false;
extern static const bool merian_restir_visibility_shade = true;
extern static const float merian_restir_boiling_filter_strength = 0.0;
}

[shader("raygeneration")]
void generate(ParameterBlock<Scene> scene, ParameterBlock<RestirBinding> params) {
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    RandomGenerator rng = restir_rng(pixel);
    RestirReservoir reservoir = restir_reservoir_init();

    const RestirSurface surface = restir_surface(scene, params.gbuffer, pixel, dim);
    if (surface.valid && surface.sp.has_surface_material) {
        const OrthonormalFrame frame = surface.sp.shading_frame;
        let bsdf = surface.sp.surface_material.get_bsdf();
        const float3 wi_local = frame.to_local(surface.wi);

        for (int s = 0; s < merian_restir_spp; s++) {
            let bs_opt = bsdf.sample_eval(wi_local, rng);
            if (!bs_opt.hasValue)
                continue;
            const ShadingFunctionSample bs = bs_opt.value;
            const float3 wo = frame.to_world(bs.wo);
            if (bs.pdf <= 0)
                continue;

            const bool transmission = wi_local.z * bs.wo.z < 0.f;
            let ray_opt = surface.sp.make_ray(surface.wi, wo, transmission);
            if (!ray_opt.hasValue)
                continue;
            const RayDesc ray = ray_opt.value;
            const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, 0xff);
            const ShadingPoint lit = scene.get_shading_point(ray, hit);
            if (all(lit.emission <= float3(0)))
                continue;

            RestirSample x;
            x.pos = lit.pos;
            x.normal = lit.get_face_normal();
            x.radiance = lit.emission;
            x.mv = float3(0);
            x.time = scene.time;
            const bool distant = restir_is_distant(scene, hit);
            x.flags = RestirSampleFlags::RestirSampleValid |
                      (distant ? RestirSampleFlags::RestirSampleDistant : 0u);

            if (!distant && merian_restir_apply_motion &&
                scene.time_diff > 0) {
                if (let gd = scene.get_geometry_data(hit)) {
                    const float3 mv =
                        (x.pos - scene.get_prev_world_position(hit, gd)) / scene.time_diff;
                    // newly spawned geometry has no valid previous position - a bogus velocity
                    // advances the sample to infinity in the temporal pass (degenerate rays hang
                    // the traversal hardware)
                    if (all(isfinite(mv)) && dot(mv, mv) < RESTIR_MAX_SAMPLE_SPEED_2) {
                        x.mv = mv;
                    }
                }
            }

            const float p_sample = bs.pdf * restir_geometry_term(x, surface.sp.pos);
            if (p_sample <= 0)
                continue;
            reservoir.add_sample(rng, x, p_sample, restir_target_pdf(x, surface, rng));
        }
    }

    reservoir.weight_sum /= float(merian_restir_spp);
    reservoir.confidence = 1.f;
    pc.reservoirs_out[restir_index(pixel, dim)] = reservoir;
}

[shader("raygeneration")]
void temporal(ParameterBlock<Scene> scene, ParameterBlock<RestirBinding> params) {
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    RandomGenerator rng = restir_rng(pixel);
    const RestirSurface surface = restir_surface(scene, params.gbuffer, pixel, dim);

    const RestirReservoir current = pc.reservoirs_in[restir_index(pixel, dim)];

    const SurfaceData cur = params.gbuffer.get_surface(pixel);
    const int2 prev_pixel = int2(round(float2(pixel) + params.gbuffer.get_motion_vectors(pixel)));
    bool reprojected = surface.valid && all(prev_pixel >= 0) && all(uint2(prev_pixel) < dim);
    if (reprojected) {
        const SurfaceData prev_surf = params.prev_gbuffer.get_surface(prev_pixel);
        reprojected = reprojection_valid(cur.normal, prev_surf.normal,
                                         pc.temporal_normal_reject_cos, cur.linear_z, cur.delta_z,
                                         prev_surf.linear_z, pc.temporal_depth_reject);
    }

    RestirReservoir prev = restir_reservoir_init();
    RestirSurface prev_surface;
    prev_surface.valid = false;
    if (reprojected) {
        prev = pc.reservoirs_prev[restir_index(uint2(prev_pixel), dim)];
        if (merian_restir_apply_motion) {
            prev.y.pos += prev.y.mv * (scene.time - prev.y.time);
            prev.y.time = scene.time;
        }
        if (pc.temporal_clamp_m > 0)
            prev.confidence = min(prev.confidence, float(pc.temporal_clamp_m));
        prev_surface = restir_surface(scene, params.prev_gbuffer, uint2(prev_pixel), dim);
    }

    // Each domain is one technique, weighed by the target it reaches there and by how many
    // samples it stands for [Lin et al. 2022, eq. 36]. Unlike a count-based correction this does
    // not care how long the history is.
    RestirReservoir reservoir = restir_reservoir_init();
    const float c_here = current.confidence;
    const float c_there = reprojected ? prev.confidence : 0.f;

    float mis_here = c_here + c_there > 0.f ? c_here / (c_here + c_there) : 1.f;
    if (current.p_target > 0.f) {
        float target_there = reprojected ? restir_target_pdf(current.y, prev_surface, rng) : 0.f;
        if (target_there > 0.f && !restir_trace_visibility(scene, prev_surface, current.y))
            target_there = 0.f;
        const float denominator = c_here * current.p_target + c_there * target_there;
        mis_here = denominator > 0.f ? (c_here * current.p_target) / denominator : 0.f;
    }
    reservoir.add(rng, mis_here, current, current.p_target);

    if (reprojected) {
        float target_here = restir_target_pdf(prev.y, surface, rng);
        if (target_here > 0.f && !restir_trace_visibility(scene, surface, prev.y))
            target_here = 0.f;
        const float denominator = c_here * target_here + c_there * prev.p_target;
        const float mis_there =
            denominator > 0.f ? (c_there * prev.p_target) / denominator : 0.f;
        reservoir.add(rng, mis_there, prev, target_here);
    }

    if (merian_restir_boiling_filter_strength > 1e-6f)
        restir_boiling_filter(reservoir, merian_restir_boiling_filter_strength);

    pc.reservoirs_out[restir_index(pixel, dim)] = reservoir;
}

[shader("raygeneration")]
void spatial(ParameterBlock<Scene> scene, ParameterBlock<RestirBinding> params) {
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    RandomGenerator rng = restir_rng(pixel);
    const RestirSurface surface = restir_surface(scene, params.gbuffer, pixel, dim);
    const SurfaceData cur = params.gbuffer.get_surface(pixel);

    const RestirReservoir central = pc.reservoirs_in[restir_index(pixel, dim)];
    const float central_target = central.p_target;
    const float central_confidence = central.confidence;

    // 1. which neighbors survive, and the confidence they carry between them. The pairwise
    // weights need that sum before the first pair is weighed, so acceptance cannot be decided
    // inside the loop below.
    const int n_iter = min(merian_restir_spatial_iterations, RESTIR_MAX_SPATIAL);
    int2 accepted_pixel[RESTIR_MAX_SPATIAL];
    float accepted_confidence[RESTIR_MAX_SPATIAL];
    int accepted = 0;
    float confidence_sum = 0.f;

    for (int i = 0; i < n_iter; i++) {
        const int2 npix =
            int2(round(float2(pixel) + pc.spatial_radius * (rng.next_float2() * 2 - 1)));
        if (any(npix < 0) || any(uint2(npix) >= dim))
            continue;
        const SurfaceData ns = params.gbuffer.get_surface(npix);
        if (!reprojection_valid(cur.normal, ns.normal, pc.spatial_normal_reject_cos, cur.linear_z,
                                0.0f, npix - int2(pixel), float2(cur.grad_z), ns.linear_z,
                                pc.spatial_depth_reject))
            continue;

        accepted_pixel[accepted] = npix;
        accepted_confidence[accepted] =
            pc.reservoirs_in[restir_index(uint2(npix), dim)].confidence;
        confidence_sum += accepted_confidence[accepted];
        accepted++;
    }

    const float total_confidence = confidence_sum + central_confidence;

    // 2. resample the pair (central, neighbor) with the defensive pairwise MIS weights of
    // [Lin et al. 2022, eq. 38] in the confidence-weighted form of [Hedstrom et al. 2026, eq. 11].
    RestirReservoir reservoir = restir_reservoir_init();
    float central_mis = total_confidence > 0.f ? central_confidence / total_confidence : 1.f;

    for (int i = 0; i < accepted; i++) {
        const uint2 npixel = uint2(accepted_pixel[i]);
        const float nconfidence = accepted_confidence[i];
        const RestirReservoir nres = pc.reservoirs_in[restir_index(npixel, dim)];
        const RestirSurface nsurf = restir_surface(scene, params.gbuffer, npixel, dim);

        // what this pixel's own sample is worth in the neighbor's domain, eq. 11b
        float target_there = 0.f;
        if (central_target > 0.f) {
            target_there = restir_target_pdf(central.y, nsurf, rng);
            if (target_there > 0.f && !restir_trace_visibility(scene, nsurf, central.y))
                target_there = 0.f;
        }
        const float central_pair =
            confidence_sum * target_there + central_confidence * central_target;
        if (central_pair > 0.f) {
            central_mis += (nconfidence / total_confidence) *
                           (central_confidence * central_target) / central_pair;
        }

        // and the neighbor's sample here, eq. 11a
        float target_here = restir_target_pdf(nres.y, surface, rng);
        if (target_here > 0.f && !restir_trace_visibility(scene, surface, nres.y))
            target_here = 0.f;
        const float pair = confidence_sum * nres.p_target + central_confidence * target_here;
        const float nmis =
            pair > 0.f ? (confidence_sum / total_confidence) * (nconfidence * nres.p_target) / pair
                       : 0.f;
        reservoir.add(rng, nmis, nres, target_here);
    }

    // the canonical sample needed the sum over the neighbors, so it goes in last
    reservoir.add(rng, central_mis, central, central_target);

    pc.reservoirs_out[restir_index(pixel, dim)] = reservoir;
}

[shader("raygeneration")]
void shade(ParameterBlock<Scene> scene, ParameterBlock<RestirBinding> params) {
    const uint2 pixel = DispatchRaysIndex().xy;
    const uint2 dim = params.gbuffer.get_dimensions();
    if (any(pixel >= dim))
        return;

    const RestirSurface surface = restir_surface(scene, params.gbuffer, pixel, dim);
    const RestirReservoir reservoir = pc.reservoirs_in[restir_index(pixel, dim)];

    float3 irradiance = float3(0);
    if (surface.valid && surface.sp.has_surface_material && restir_sample_valid(reservoir.y) &&
        isfinite(reservoir.get_ucw())) {
        const float3 to_light = reservoir.y.pos - surface.sp.pos;
        const float dist2 = dot(to_light, to_light);
        const float3 wo = to_light * rsqrt(dist2);
        const OrthonormalFrame frame = surface.sp.shading_frame;
        const float3 wi_local = frame.to_local(surface.wi);
        const float3 wo_local = frame.to_local(wo);

        const bool visible = !merian_restir_visibility_shade ||
                             restir_trace_visibility(scene, surface, reservoir.y);

        if (wo_local.z > 0 && wi_local.z > 0 && visible) {
            RandomGenerator rng = restir_rng(pixel);
            const float3 f_cos =
                surface.sp.surface_material.get_bsdf().eval(wi_local, wo_local, rng);
            const float geo = (reservoir.y.flags & RestirSampleFlags::RestirSampleDistant) != 0
                                  ? 1.0f
                                  : max(dot(reservoir.y.normal, -wo), 0.0f) / dist2;
            irradiance = f_cos * reservoir.y.radiance * reservoir.get_ucw() * geo;
        }
    }

    if (merian_render_emission_on_primary) {
        const RayDesc ray = scene.generate_camera_ray(pixel, dim);
        const ShadingPoint primary = scene.get_shading_point(ray, params.gbuffer.get_hit(pixel));
        irradiance += primary.emission;
    }

    if (merian_render_demodulate_albedo) {
        irradiance /= params.gbuffer.get_demodulation_albedo(pixel);
    }

    const float l = yuv_luminance(irradiance);
    params.irradiance[pixel] = float4(irradiance, l * l);
}
