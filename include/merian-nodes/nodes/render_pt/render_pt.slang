import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.colors.colorspaces;

using merian;

namespace merian {
extern static const bool merian_render_emission_on_primary = false;
}

struct RenderBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
};

struct PushConstant {
    int spp;
    int max_path_length;
    uint instance_mask;
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

float3 trace_path(const ParameterBlock<Scene> scene,
                  RayDesc ray,
                  ShadingPoint sp,
                  inout RandomGenerator rng) {
    float3 radiance = merian_render_emission_on_primary ? sp.material.emission : float3(0);
    float3 throughput = float3(1);

    for (int bounce = 1; bounce < pc.max_path_length; bounce++) {
        const OrthonormalFrame frame = sp.material.shading_frame;
        let bsdf = sp.material.get_bsdf();
        const float3 wi_local = frame.to_local(-ray.Direction);

        let sample = bsdf.sample_eval(wi_local, rng);
        if (!sample.hasValue)
            break;
        const ShadingFunctionSample bs = sample.value;

        const float3 wo = frame.to_world(bs.wo);
        throughput *= bs.weight;

        // avoid self-intersection by nudging off the surface on wo's side
        const float3 ng = sp.get_face_normal();
        ray.Origin = sp.pos + (dot(wo, ng) < 0.f ? -ng : ng) * SCENE_RAY_OFFSET;
        ray.Direction = wo;
        ray.TMin = SCENE_RAY_TMIN;
        ray.TMax = SCENE_RAY_TMAX;

        sp = scene.trace_and_get_shading_point(ray, pc.instance_mask, uint16_t(bounce));
        radiance += throughput * sp.material.emission;
    }

    return radiance;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID,
          ParameterBlock<Scene> scene,
          ParameterBlock<RenderBinding> params) {

    const uint2 pixel = tid.xy;
    if (not_on_texture(pixel, params.irradiance))
        return;

    const uint2 dim = params.gbuffer.get_dimensions();
    const Hit primary_hit = params.gbuffer.get_hit(pixel);

    // primary ray and shading point are identical for every sample
    const RayDesc primary_ray = scene.camera.generate_ray_pinhole(pixel, dim);
    const ShadingPoint primary_sp = scene.get_shading_point(primary_ray, primary_hit);

    float3 irradiance = float3(0);
    float second_moment = 0.f;

    for (int s = 0; s < pc.spp; s++) {
        RandomGenerator rng = XorShift32(pcg4d16(uint4(pixel, scene.frame, s)));

        const float3 contrib = trace_path(scene, primary_ray, primary_sp, rng);
        if (!any(isnan(contrib)) && !any(isinf(contrib))) {
            irradiance += contrib;
            const float l = yuv_luminance(contrib);
            second_moment += l * l;
        }
    }

    if (pc.spp > 0) {
        irradiance /= float(pc.spp);
        second_moment /= float(pc.spp);
    }

    params.irradiance[pixel] = float4(irradiance, second_moment);
}
