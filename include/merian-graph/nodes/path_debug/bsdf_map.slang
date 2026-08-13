import merian_shaders.scene.camera;
import merian_shaders.scene.scene;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.utils.encoding;
import merian_shaders.utils.frame;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

ParameterBlock<GBuffer> gbuffer;

// Evaluates the BSDF pdf at the selected pixel's primary hit for every map bin — the analytic
// reference the sampled-direction density is compared against. Bin directions use the same
// gbuffer-normal frame as the accumulation so the maps align even for normal-mapped materials.
[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID, ParameterBlock<Scene> scene) {
    const uint2 cell = gid.xy;
    if (any(cell >= params.map_res)) {
        return;
    }
    const uint bin = cell.y * params.map_res + cell.x;

    const Hit hit = gbuffer.get_hit(path_debug_selected_pixel());
    const RayDesc ray = scene.generate_camera_ray(path_debug_selected_pixel(), path_debug_dim());
    const ShadingPoint sp = scene.get_shading_point(ray, hit);
    if (!sp.has_surface_material) {
        // the selected pixel sees the environment: no pdf reference
        maps[path_debug_map_pdf_at(bin)] = asuint(0.f);
        return;
    }

    float3x3 to_map = dir_map_frame(float3(0, 0, 1), false);
    if (params.map_frame == 1) {
        to_map = dir_map_frame(gbuffer.get_normal(path_debug_selected_pixel()), true);
    }

    // Bin-average pdf over 4x4 sub-bin directions: bin-center evaluation misestimates bins
    // where the pdf varies steeply (horizon, sharp lobes) and would fake bias in the z-score.
    // Bins the sub-samples cannot resolve are stored with a negative sign and excluded from the
    // chi-square test; displays use the absolute value.
    const OrthonormalFrame frame = sp.shading_frame;
    let bsdf = sp.surface_material.get_bsdf();
    const float3 wi_local = frame.to_local(-ray.Direction);
    float pdf_sum = 0.f;
    float pdf_min = 1e30f;
    float pdf_max = 0.f;
    uint valid = 0;
    for (uint qy = 0; qy < 4; qy++) {
        for (uint qx = 0; qx < 4; qx++) {
            const float2 uv =
                (float2(cell) + float2(0.125f + 0.25f * qx, 0.125f + 0.25f * qy)) /
                float(params.map_res);
            const float4 map_dir = dir_map_dir(params.map_transform, uv);
            if (map_dir.w == 0.f) {
                continue;
            }
            const float3 wo_world = mul(transpose(to_map), map_dir.xyz);
            float pdf = bsdf.pdf(wi_local, frame.to_local(wo_world));
            if (isnan(pdf) || isinf(pdf)) {
                pdf = 0.f;
            }
            pdf_sum += pdf;
            pdf_min = min(pdf_min, pdf);
            pdf_max = max(pdf_max, pdf);
            valid++;
        }
    }
    float avg = valid > 0 ? pdf_sum / float(valid) : 0.f;
    // steep variation or partial domain coverage: no reliable expectation for this bin
    if (valid < 16 || pdf_max > 8.f * pdf_min + 1e-3f) {
        avg = -avg;
    }
    maps[path_debug_map_pdf_at(bin)] = asuint(avg);
}

