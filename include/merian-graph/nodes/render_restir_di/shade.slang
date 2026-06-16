#include "merian-graph/nodes/render_restir_di/render_restir_di.slangh"

using merian;

// Final shading: evaluate the selected light sample weighted by W and write the irradiance.
[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<RestirBinding> params) {
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

        const bool visible = (pc.flags & RestirFlags::RestirFlagVisibilityShade) == 0 ||
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

    const float l = yuv_luminance(irradiance);
    params.irradiance[pixel] = float4(irradiance, l * l);
}
