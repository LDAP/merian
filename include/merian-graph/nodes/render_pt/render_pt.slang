import merian_shaders.gbuffer;
import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import merian_shaders.colors.colorspaces;
__exported import merian_shaders.sampling.guiding;

// Selects the guiding method at link time; render_pt.cpp injects the module.
__exported import render_pt_guiding;

using merian;

namespace merian {
extern static const bool merian_render_emission_on_primary = true;
extern static const int merian_render_spp = 1;
extern static const int merian_render_max_path_length = 8;
extern static const uint merian_render_instance_mask = 0xFFu;
extern static const bool merian_render_enable_ser = false;
extern static const bool merian_render_demodulate_albedo = false;
}

struct RenderBinding {
    ParameterBlock<GBuffer> gbuffer;
    RWTexture2D<float4> irradiance;
    RenderGuiding guiding;
}

float3 trace_path(const ParameterBlock<Scene> scene,
                  const ParameterBlock<RenderBinding> params,
                  RayDesc ray,
                  ShadingPoint sp,
                  const uint16_t sample_index,
                  inout RandomGenerator rng) {
    // Medium the ray is travelling through; updated on transmission events.
    HomogeneousVolume medium = scene.exterior_volume;

    float3 throughput = medium.transmittance(distance(ray.Origin, sp.pos));
    float3 radiance = merian_render_emission_on_primary ? throughput * sp.emission : float3(0);

    const RenderFrameInfo frame_info = {scene.camera.position, scene.prev_camera.position,
                                        scene.time, scene.time_diff, scene.frame};
    var guiding_state = params.guiding.begin_path();

    for (uint16_t bounce = 1; bounce < merian_render_max_path_length; bounce++) {
        // the ray left the scene: nothing to scatter from, emission is already accumulated
        if (!sp.has_surface_material || all(throughput < SCENE_THROUGHPUT_THRESHOLD)) {
            break;
        }

        const OrthonormalFrame frame = sp.shading_frame;
        let bsdf = sp.surface_material.get_bsdf();
        const float3 wi_local = frame.to_local(-ray.Direction);
        const PathVertex vertex = {sp.pos,     sp.prev_pos, frame,
                                   bsdf.alpha, bounce,      sample_index};

        // 1. the guiding lobes around this vertex, mixed with the BSDF
        let guiding = params.guiding.get_distribution(guiding_state, frame_info, vertex, rng);
        const ScatterMixture mixture = ScatterMixture(guiding, bsdf.alpha);

        // 2. one scatter direction out of the mixture
        const Optional<ScatterSample> scatter = mixture.sample(bsdf, guiding, frame, wi_local, rng);
        if (!scatter.hasValue) {
            break;
        }
        const ScatterSample s = scatter.value;
        const float3 wo_local = frame.to_local(s.wo);

        // resolved before the continuation ray so that no candidate stays live across it
        PathSegment segment;
        segment.pos = vertex.pos;
        segment.normal = frame.get_normal();
        segment.bounce = bounce;
        segment.wo = s.wo;
        segment.pdf = s.pdf;
        segment.weight = s.weight;
        segment.delta = s.delta;
        segment.traced = false;
        segment.is_last = bounce + uint16_t(1) >= merian_render_max_path_length;
        segment.next_pos = vertex.pos;
        segment.next_normal = segment.normal;
        segment.next_mv = half3(0);
        segment.incident = float3(0);

        // Last thing before the ray: the record is what crosses the traversal, so it must not be
        // taken while the mixture densities above are still live.
        let guiding_record = params.guiding.record_scatter(guiding, s.guided);

#if defined(MERIAN_SPIRV_CAP_SUPPORTED_ShaderInvocationReorderEXT)
        if (merian_render_enable_ser) {
            ReorderThread(0, 0);
        }
#endif

        // A transmission event crosses the surface; update the medium we are now in.
        const bool transmission = wi_local.z * wo_local.z < 0.f;
        Optional<RayDesc> next_ray = none;
        if (any(s.weight > 0.f)) {
            next_ray = sp.make_ray(-ray.Direction, s.wo, transmission);
        }

        if (let next = next_ray) {
            if (transmission) {
                const IsotropicVolume interior = sp.surface_material.get_interior_volume();
                // leaving a surface puts the ray back into the medium enclosing the scene
                if (dot(-ray.Direction, sp.get_face_normal()) >= 0.f)
                    medium = interior;
                else
                    medium = scene.exterior_volume;
            }
            ray = next;

            // the last vertex is never scattered from, so its material sample would be dead
            sp = scene.trace_and_get_shading_point(ray, merian_render_instance_mask,
                                                   RenderContext(bounce), DefaultLODSampler(),
                                                   !segment.is_last);

            // Beer-Lambert attenuation over the segment just travelled inside the medium.
            // In-scattering is not accounted for here; that is the volume renderer's job. Folded
            // into the incident radiance so that the guiding sees the attenuated signal.
            const float3 segment_transmittance =
                medium.transmittance(distance(ray.Origin, sp.pos));

            segment.traced = true;
            segment.next_pos = sp.pos;
            segment.next_normal = sp.shading_normal;
            segment.next_mv = half3((sp.pos - sp.prev_pos) / scene.time_diff);
            segment.incident = sp.emission * segment_transmittance;

            throughput *= segment.weight;
            radiance += throughput * segment.incident;
        }

        // 3. what the guiding learns, and the estimate it completes the path with
        const float3 tail =
            params.guiding.record_segment(guiding_state, frame_info, guiding_record, segment, rng);
        radiance += throughput * tail;

        if (!next_ray.hasValue) {
            break;
        }
    }

    return radiance;
}

void render_pixel(const uint2 pixel,
                  const ParameterBlock<Scene> scene,
                  const ParameterBlock<RenderBinding> params) {
    if (not_on_texture(pixel, params.irradiance))
        return;

    const uint2 dim = params.gbuffer.get_dimensions();
    const Hit primary_hit = params.gbuffer.get_hit(pixel);

    // primary ray and shading point are identical for every sample
    const RayDesc primary_ray = scene.generate_camera_ray(pixel, dim);
    const ShadingPoint primary_sp = scene.get_shading_point(primary_ray, primary_hit);

    // demodulate per sample so the second moment stays consistent with the stored radiance
    const float3 demod_albedo = merian_render_demodulate_albedo
                                    ? params.gbuffer.get_demodulation_albedo(pixel)
                                    : float3(1);

    float3 irradiance = float3(0);
    float second_moment = 0.f;

    for (int s = 0; s < merian_render_spp; s++) {
        RandomGenerator rng = PCG(pcg4d16(uint4(pixel, scene.frame, s)));

        const float3 contrib =
            trace_path(scene, params, primary_ray, primary_sp, uint16_t(s), rng) / demod_albedo;
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

// Two entry points over the same body: compute is faster (no RT-pipeline VGPR cap), while the
// raygen variant is available for experiments that need ray-tracing-pipeline-only features (SER).
[shader("compute")]
[numthreads(8, 8, 1)]
void main_compute(uint3 thread_id: SV_DispatchThreadID,
                  ParameterBlock<Scene> scene,
                  ParameterBlock<RenderBinding> params) {
    render_pixel(thread_id.xy, scene, params);
}

[shader("raygeneration")]
void main(ParameterBlock<Scene> scene, ParameterBlock<RenderBinding> params) {
    render_pixel(DispatchRaysIndex().xy, scene, params);
}
