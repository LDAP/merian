#include "merian-graph/nodes/render_restir_di/render_restir_di.slangh"

using merian;

// Spatial reuse: resample a set of nearby reservoirs from the input buffer.
[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<RestirBinding> params) {
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

    const int n_iter = min(pc.spatial_iterations, RESTIR_MAX_SPATIAL);
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

    if (pc.spatial_bias_correction == RestirBiasCorrection::RestirBiasNone) {
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
                pc.spatial_bias_correction == RestirBiasCorrection::RestirBiasRaytraced &&
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
