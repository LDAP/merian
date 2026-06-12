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
    // Medium the ray is travelling through; updated on transmission events.
    float current_ior = 1.0f;
    float3 current_absorption = float3(0.0f);

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

        const float3 ng = sp.get_face_normal();
        if (merian_hint_enable_transmission) {
            // A transmission event crosses the geometric surface; update the medium we are now in.
            if ((dot(wo, ng) < 0.f) != (dot(-ray.Direction, ng) < 0.f)) {
                current_ior /= sp.material.get_eta();
                if (merian_hint_enable_volume) {
                    current_absorption = sp.material.get_absorption();
                }
            }
        }

        // avoid self-intersection by nudging off the surface on wo's side
        ray.Origin = sp.pos + (dot(wo, ng) < 0.f ? -ng : ng) * SCENE_RAY_OFFSET;
        ray.Direction = wo;
        ray.TMin = SCENE_RAY_TMIN;
        ray.TMax = SCENE_RAY_TMAX;

        const Hit hit = scene.trace_ray(ray, RAY_FLAG_NONE, pc.instance_mask);
        sp = scene.get_shading_point(ray, hit, RenderContext(uint16_t(bounce), current_ior));

        if (merian_hint_enable_volume) {
            // Beer-Lambert absorption over the segment just travelled inside the medium.
            throughput *= exp(-current_absorption * distance(ray.Origin, sp.pos));
        }
        radiance += throughput * sp.material.emission;
    }

    return radiance;
}

[shader("raygeneration")]
void main(ParameterBlock<Scene> scene,
          ParameterBlock<RenderBinding> params) {

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
