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

struct RenderBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
    RWTexture2D<float4> debug;
    HashedIrradianceCache<lc_buffer_size, lc_probe_count, lc_stochastic_interpolation> irr_cache;
    MCPG<mc_adaptive_buffer_size> mcpg;
};

float3 trace_path(const ParameterBlock<Scene> scene,
                  RayDesc ray,
                  ShadingPoint sp,
                  inout RandomGenerator rng) {
    float3 radiance = merian_render_emission_on_primary ? sp.material.emission : float3(0);

    float current_ior = 1.0f;
    float3 throughput = float3(1);
    float3 f = float3(0);
    float p = 1.0;

    // TODO: Volume absoption support.
    // float3 current_absorption = float3(0.0f);

    for (int bounce = 1; bounce < merian_render_max_path_length; bounce++) {
        const OrthonormalFrame frame = sp.material.shading_frame;
        let bsdf = sp.material.get_bsdf();
        const float3 wi_local = frame.to_local(-ray.Direction);

        let sample = bsdf.sample_eval(wi_local, rng);
        if (!sample.hasValue)
            break;
        const ShadingFunctionSample bs = sample.value;

        const float3 wo = frame.to_world(bs.wo);
        throughput *= bs.weight;

        const float3 ng = sp.get_face_normal();
        // A transmission event crosses the geometric surface; update the medium we are now in.
        if ((dot(wo, ng) < 0.f) != (dot(-ray.Direction, ng) < 0.f)) {
            current_ior /= sp.material.get_eta();
            // TODO: Vulume absoption support
            // current_absorption = sp.material.get_absorption();
        }

        // avoid self-intersection by nudging off the surface on wo's side
        ray.Origin = sp.pos + (dot(wo, ng) < 0.f ? -ng : ng) * SCENE_RAY_OFFSET;
        ray.Direction = wo;
        ray.TMin = SCENE_RAY_TMIN;
        ray.TMax = SCENE_RAY_TMAX;

        const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, merian_render_instance_mask);
        sp = scene.get_shading_point(ray, hit, RenderContext(uint16_t(bounce), current_ior));

        // TODO: Vulume absoption support
        // Beer-Lambert absorption over the segment just travelled inside the medium.
        // if (any(current_absorption != float3(0.f))) {
        //     throughput *= exp(-current_absorption * distance(ray.Origin, sp.pos));
        // }

        radiance += throughput * sp.material.emission;
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
        const float3 contrib = trace_path(scene, primary_ray, primary_sp, rng);

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
    case 1: // moments
        params.debug[pixel] = float4(yuv_luminance(irradiance), second_moment.x, 0, 1);
        break;
    case 2: // light cache
        params.debug[pixel] =
            float4(params.irr_cache.get(primary_sp.pos, primary_sp.get_face_normal(), cam_pos,
                                        prev_cam_pos, rng),
                   1);
        break;
    case 3: // mc grid
        const uint level = params.mcpg.grid.distributed_level_for_pos(primary_sp.pos, cam_pos, rng);
        const int3 gidx = params.mcpg.grid.grid_idx(
            primary_sp.pos, params.mcpg.grid.grid_width_for_level(level), rng);
        params.debug[pixel] = float4(random_color(params.mcpg.grid.buffer_index(gidx, level)), 1);
        break;
    case 4: // mc grid lod (level)
        params.debug[pixel] =
            params.mcpg.grid.distributed_level_for_pos(primary_sp.pos, cam_pos, rng) / 63.0;
        break;
    default: // irradiance
        params.debug[pixel] = float4(irradiance, 1);
        break;
    }
}
