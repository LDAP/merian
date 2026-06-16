#include "merian-graph/nodes/render_restir_di/render_restir_di.slangh"

using merian;

// Temporal reuse: reproject through the motion vector and merge the previous frame's reservoir.
[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<RestirBinding> params) {
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

        if ((pc.flags & RestirFlags::RestirFlagApplyMotion) != 0) {
            prev.y.pos += prev.y.mv * (scene.time - prev.y.time);
            prev.y.time = scene.time;
        }
        if (pc.temporal_clamp_m > 0)
            prev.M = min(prev.M, uint(pc.temporal_clamp_m));

        combined_prev = reservoir.combine(rng, prev, restir_target_pdf(prev.y, surface, rng));
    }

    if (pc.temporal_bias_correction == RestirBiasCorrection::RestirBiasNone) {
        reservoir.finalize();
    } else {
        float pi = reservoir.p_target;
        float pi_sum = reservoir.p_target * current.M;
        if (reprojected) {
            const RestirSurface prev_surface =
                restir_surface(scene, params.prev_gbuffer, uint2(prev_pixel), dim);
            float temporal_p = restir_target_pdf(reservoir.y, prev_surface, rng);
            if (temporal_p > 0 &&
                pc.temporal_bias_correction == RestirBiasCorrection::RestirBiasRaytraced &&
                !restir_trace_visibility(scene, surface, reservoir.y)) {
                temporal_p = 0;
            }
            if (combined_prev)
                pi = temporal_p;
            pi_sum += temporal_p * prev.M;
        }
        reservoir.finalize(pi, pi_sum);
    }

    if (pc.boiling_filter_strength > 1e-6f)
        restir_boiling_filter(reservoir, pc.boiling_filter_strength);

    pc.reservoirs_out[restir_index(pixel, dim)] = reservoir;
}
