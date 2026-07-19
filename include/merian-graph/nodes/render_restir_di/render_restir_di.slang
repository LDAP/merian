#include "merian-graph/nodes/render_restir_di/render_restir_di.slangh"

using merian;

namespace merian {
extern static const bool merian_render_emission_on_primary = true;
extern static const int merian_restir_spp = 4;
extern static const int merian_restir_spatial_iterations = 1;
extern static const int merian_restir_temporal_bias_correction = 2;
extern static const int merian_restir_spatial_bias_correction = 1;
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
    if (surface.valid) {
        const OrthonormalFrame frame = surface.sp.material.shading_frame;
        let bsdf = surface.sp.material.get_bsdf();
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
            if (all(lit.material.emission <= float3(0)))
                continue;

            RestirSample x;
            x.pos = lit.pos;
            x.normal = lit.get_face_normal();
            x.radiance = lit.material.emission;
            x.mv = float3(0);
            x.time = scene.time;
            const bool distant = restir_is_distant(scene, hit);
            x.flags = RestirSampleFlags::RestirSampleValid |
                      (distant ? RestirSampleFlags::RestirSampleDistant : 0u);

            if (!distant && merian_restir_apply_motion &&
                scene.time_diff > 0) {
                if (let gd = scene.get_geometry_data(hit)) {
                    x.mv = (x.pos - scene.get_prev_world_position(hit, gd)) / scene.time_diff;
                }
            }

            const float p_sample = bs.pdf * restir_geometry_term(x, surface.sp.pos);
            if (p_sample <= 0)
                continue;
            reservoir.add_sample(rng, x, p_sample, restir_target_pdf(x, surface, rng));
        }
    }

    reservoir.finalize(1.0f, float(merian_restir_spp));
    reservoir.M = 1;
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

    RestirReservoir reservoir = restir_reservoir_init();
    const RestirReservoir current = pc.reservoirs_in[restir_index(pixel, dim)];
    reservoir.combine(rng, current, current.p_target);

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
    bool combined_prev = false;
    if (reprojected) {
        prev = pc.reservoirs_prev[restir_index(uint2(prev_pixel), dim)];

        if (merian_restir_apply_motion) {
            prev.y.pos += prev.y.mv * (scene.time - prev.y.time);
            prev.y.time = scene.time;
        }
        if (pc.temporal_clamp_m > 0)
            prev.M = min(prev.M, uint(pc.temporal_clamp_m));

        combined_prev = reservoir.combine(rng, prev, restir_target_pdf(prev.y, surface, rng));
    }

    if (merian_restir_temporal_bias_correction == RestirBiasCorrection::RestirBiasNone) {
        reservoir.finalize();
    } else {
        float pi = reservoir.p_target;
        float pi_sum = reservoir.p_target * current.M;
        if (reprojected) {
            const RestirSurface prev_surface =
                restir_surface(scene, params.prev_gbuffer, uint2(prev_pixel), dim);
            float temporal_p = restir_target_pdf(reservoir.y, prev_surface, rng);
            if (temporal_p > 0 &&
                merian_restir_temporal_bias_correction == RestirBiasCorrection::RestirBiasRaytraced &&
                !restir_trace_visibility(scene, surface, reservoir.y)) {
                temporal_p = 0;
            }
            if (combined_prev)
                pi = temporal_p;
            pi_sum += temporal_p * prev.M;
        }
        reservoir.finalize(pi, pi_sum);
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

    RestirReservoir reservoir = restir_reservoir_init();
    const RestirReservoir current = pc.reservoirs_in[restir_index(pixel, dim)];
    reservoir.combine(rng, current, current.p_target);

    const int n_iter = min(merian_restir_spatial_iterations, RESTIR_MAX_SPATIAL);
    int2 neighbors[RESTIR_MAX_SPATIAL];
    for (int i = 0; i < RESTIR_MAX_SPATIAL; i++)
        neighbors[i] = int2(-1);

    int selected = -1;
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

        neighbors[i] = npix;
        const RestirReservoir nr = pc.reservoirs_in[restir_index(uint2(npix), dim)];
        if (reservoir.combine(rng, nr, restir_target_pdf(nr.y, surface, rng)))
            selected = i;
    }

    if (merian_restir_spatial_bias_correction == RestirBiasCorrection::RestirBiasNone) {
        reservoir.finalize();
    } else {
        float pi = reservoir.p_target;
        float pi_sum = reservoir.p_target * current.M;
        for (int i = 0; i < n_iter; i++) {
            if (any(neighbors[i] < 0))
                continue;
            const RestirSurface nsurf =
                restir_surface(scene, params.gbuffer, uint2(neighbors[i]), dim);
            float spatial_p = restir_target_pdf(reservoir.y, nsurf, rng);
            if (spatial_p > 0 &&
                merian_restir_spatial_bias_correction == RestirBiasCorrection::RestirBiasRaytraced &&
                !restir_trace_visibility(scene, nsurf, reservoir.y)) {
                spatial_p = 0;
            }
            if (selected == i)
                pi = spatial_p;
            pi_sum += spatial_p * pc.reservoirs_in[restir_index(uint2(neighbors[i]), dim)].M;
        }
        reservoir.finalize(pi, pi_sum);
    }

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
    if (surface.valid && restir_sample_valid(reservoir.y) && !isnan(reservoir.w_sum_or_W) &&
        !isinf(reservoir.w_sum_or_W)) {
        const float3 to_light = reservoir.y.pos - surface.sp.pos;
        const float dist2 = dot(to_light, to_light);
        const float3 wo = to_light * rsqrt(dist2);
        const OrthonormalFrame frame = surface.sp.material.shading_frame;
        const float3 wi_local = frame.to_local(surface.wi);
        const float3 wo_local = frame.to_local(wo);

        const bool visible = !merian_restir_visibility_shade ||
                             restir_trace_visibility(scene, surface, reservoir.y);

        if (wo_local.z > 0 && wi_local.z > 0 && visible) {
            RandomGenerator rng = restir_rng(pixel);
            const float3 f_cos = surface.sp.material.get_bsdf().eval(wi_local, wo_local, rng);
            const float geo = (reservoir.y.flags & RestirSampleFlags::RestirSampleDistant) != 0
                                  ? 1.0f
                                  : max(dot(reservoir.y.normal, -wo), 0.0f) / dist2;
            irradiance = f_cos * reservoir.y.radiance * reservoir.w_sum_or_W * geo;
        }
    }

    if (merian_render_emission_on_primary) {
        const RayDesc ray = scene.generate_camera_ray(pixel, dim);
        const ShadingPoint primary = scene.get_shading_point(ray, params.gbuffer.get_hit(pixel));
        irradiance += primary.material.emission;
    }

    const float l = yuv_luminance(irradiance);
    params.irradiance[pixel] = float4(irradiance, l * l);
}
