import merian_shaders.scene.camera;
import merian_shaders.scene.scene;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;
import merian_shaders.utils.frame;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

ParameterBlock<GBuffer> gbuffer;

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID, ParameterBlock<Scene> scene) {
    const uint2 cell = gid.xy;
    if (any(cell >= params.map_res)) {
        return;
    }
    const uint bin = cell.y * params.map_res + cell.x;

    const Hit hit = gbuffer.get_hit(path_debug_picked_pixel());
    const RayDesc ray = scene.generate_camera_ray(path_debug_picked_pixel(), path_debug_dim());
    const ShadingPoint sp = scene.get_shading_point(ray, hit);
    if (!sp.has_surface_material) {
        maps[path_debug_map_pdf_at(bin)] = asuint(0.f);
        maps[path_debug_map_eval_at(bin)] = asuint(0.f);
        return;
    }

    const float3x3 to_map = dir_map_frame(gbuffer.get_normal(path_debug_picked_pixel()),
                                          params.map_frame == PATH_DEBUG_FRAME_LOCAL);

    // mean over 4x4 sub-bin directions; negative marks an unresolved bin
    const OrthonormalFrame frame = sp.shading_frame;
    let bsdf = sp.surface_material.get_bsdf();
    const float3 wi_local = path_debug_wi_local(frame.to_local(-ray.Direction));
    RandomGenerator rng = PCG(xxhash32(uint4(cell, 17u, 71u)));
    float pdf_sum = 0.f;
    float eval_sum = 0.f;
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
            const float3 wo_local = frame.to_local(wo_world);
            float pdf = bsdf.pdf(wi_local, wo_local);
            if (isnan(pdf) || isinf(pdf)) {
                pdf = 0.f;
            }
            float eval_lum = yuv_luminance(bsdf.eval(wi_local, wo_local, rng));
            if (isnan(eval_lum) || isinf(eval_lum)) {
                eval_lum = 0.f;
            }
            eval_sum += eval_lum;
            pdf_sum += pdf;
            pdf_min = min(pdf_min, pdf);
            pdf_max = max(pdf_max, pdf);
            valid++;
        }
    }
    float avg = valid > 0 ? pdf_sum / float(valid) : 0.f;
    if (valid < 16 || pdf_max > 8.f * pdf_min + 1e-3f) {
        avg = -avg;
    }
    maps[path_debug_map_pdf_at(bin)] = asuint(avg);
    maps[path_debug_map_eval_at(bin)] = asuint(valid > 0 ? eval_sum / float(valid) : 0.f);
}
