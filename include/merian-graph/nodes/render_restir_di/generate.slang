#include "merian-graph/nodes/render_restir_di/render_restir_di.slangh"

using merian;

// Initial candidates by BSDF sampling.
[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<RestirBinding> params) {
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
        const float3 ng = surface.sp.get_face_normal();

        for (int s = 0; s < pc.spp; s++) {
            let bs_opt = bsdf.sample_eval(wi_local, rng);
            if (!bs_opt.hasValue)
                continue;
            const ShadingFunctionSample bs = bs_opt.value;
            const float3 wo = frame.to_world(bs.wo);
            if (bs.pdf <= 0 || dot(wo, ng) <= 1e-3f)
                continue;

            RayDesc ray;
            ray.Origin = surface.sp.pos + ng * SCENE_RAY_OFFSET;
            ray.Direction = wo;
            ray.TMin = SCENE_RAY_TMIN;
            ray.TMax = SCENE_RAY_TMAX;
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

            if (!distant && (pc.flags & RestirFlags::RestirFlagApplyMotion) != 0 &&
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

    reservoir.finalize();
    pc.reservoirs_out[restir_index(pixel, dim)] = reservoir;
}
