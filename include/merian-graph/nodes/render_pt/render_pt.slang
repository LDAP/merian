import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.colors.colorspaces;

using merian;

namespace merian {
extern static const bool merian_render_emission_on_primary = true;
extern static const int merian_render_spp = 1;
extern static const int merian_render_max_path_length = 8;
extern static const uint merian_render_instance_mask = 0xFFu;
extern static const bool merian_render_enable_ser = false;
}

struct RenderBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
};

float3 trace_path(const ParameterBlock<Scene> scene,
                  RayDesc ray,
                  ShadingPoint sp,
                  inout RandomGenerator rng) {
    float3 radiance = merian_render_emission_on_primary ? sp.material.emission : float3(0);
    float3 throughput = float3(1);
    // Medium the ray is travelling through; updated on transmission events.
    float current_ior = 1.0f;
    float3 current_absorption = float3(0.0f);

    for (int bounce = 1; bounce < merian_render_max_path_length; bounce++) {
        if (all(throughput < SCENE_THROUGHPUT_THRESHOLD)) {
            break;
        }

        const OrthonormalFrame frame = sp.material.shading_frame;
        let bsdf = sp.material.get_bsdf();
        const float3 wi_local = frame.to_local(-ray.Direction);
        let sample = bsdf.sample_eval(wi_local, rng);
        if (!sample.hasValue) {
            break;
        }

#if defined(MERIAN_SPIRV_CAP_SUPPORTED_ShaderInvocationReorderEXT)
        if (merian_render_enable_ser) {
            ReorderThread(0, 0);
        }
#endif

        const ShadingFunctionSample bs = sample.value;
        const float3 wo = frame.to_world(bs.wo);
        throughput *= bs.weight;

        const float3 ng = sp.get_face_normal();
        // A transmission event crosses the geometric surface; update the medium we are now in.
        if ((dot(wo, ng) < 0.f) != (dot(-ray.Direction, ng) < 0.f)) {
            current_ior /= sp.material.get_eta();
            current_absorption = sp.material.get_absorption();
        }

        // avoid self-intersection by nudging off the surface on wo's side
        ray.Origin = sp.pos + (dot(wo, ng) < 0.f ? -ng : ng) * SCENE_RAY_OFFSET;
        ray.Direction = wo;
        ray.TMin = SCENE_RAY_TMIN;
        ray.TMax = SCENE_RAY_TMAX;

        sp = scene.trace_and_get_shading_point(ray, merian_render_instance_mask,
                                               RenderContext(uint16_t(bounce), current_ior));

        // Beer-Lambert absorption over the segment just travelled inside the medium.
        if (any(current_absorption != float3(0.f))) {
            throughput *= exp(-current_absorption * distance(ray.Origin, sp.pos));
        }

        radiance += throughput * sp.material.emission;
    }

    return radiance;
}

[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<RenderBinding> params) {
    const uint2 pixel = DispatchRaysIndex().xy;
    if (not_on_texture(pixel, params.irradiance))
        return;

    const uint2 dim = params.gbuffer.get_dimensions();
    const Hit primary_hit = params.gbuffer.get_hit(pixel);

    // primary ray and shading point are identical for every sample
    const RayDesc primary_ray = scene.generate_camera_ray(pixel, dim);
    const ShadingPoint primary_sp = scene.get_shading_point(primary_ray, primary_hit);

    float3 irradiance = float3(0);
    float second_moment = 0.f;

    for (int s = 0; s < merian_render_spp; s++) {
        RandomGenerator rng = PCG(pcg4d16(uint4(pixel, scene.frame, s)));

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
}
