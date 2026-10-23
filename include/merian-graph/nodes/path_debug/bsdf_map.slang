import merian_shaders.scene.camera;
import merian_shaders.scene.scene;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.utils.dir_map;
import merian_shaders.utils.frame;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

ParameterBlock<GBuffer> gbuffer;

// directions per bin side the pdf and eval are averaged over
static const uint SUB_BINS = 4;

[numthreads(PATH_DEBUG_WG_2D, PATH_DEBUG_WG_2D, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID, ParameterBlock<Scene> scene) {
    const uint2 cell = gid.xy;
    if (any(cell >= params.map_res)) {
        return;
    }
    const uint bin = path_debug_map_bin(cell);

    const Hit hit = gbuffer.get_hit(path_debug_selected_pixel());
    const RayDesc ray = scene.generate_camera_ray(path_debug_selected_pixel(), path_debug_dim());
    const ShadingPoint sp = scene.get_shading_point(ray, hit);
    if (!sp.has_surface_material) {
        maps[path_debug_map_pdf_at(bin)] = asuint(0.f);
        path_debug_store_float3(maps, path_debug_map_eval_at(bin), float3(0.f));
        return;
    }

    const float3 selected_normal = gbuffer.get_first_hit_normal(path_debug_selected_pixel());
    const float3x3 to_map = dir_map_frame(path_debug_map_normal(selected_normal),
                                          params.map_frame == PATH_DEBUG_FRAME_LOCAL);

    // the mean over the sub-bin directions; negative marks an unresolved bin
    const OrthonormalFrame frame = sp.shading_frame;
    let bsdf = sp.surface_material.get_bsdf();
    const float3 wi_local = path_debug_wi_local(frame.to_local(-ray.Direction));
    RandomGenerator rng = PCG(xxhash32(uint4(cell, 17u, 71u)));
    float pdf_sum = 0.f;
    float3 eval_sum = float3(0.f);
    float pdf_min = 1e30f;
    float pdf_max = 0.f;
    uint valid = 0;
    for (uint qy = 0; qy < SUB_BINS; qy++) {
        for (uint qx = 0; qx < SUB_BINS; qx++) {
            const float2 uv = (float2(cell) + (float2(qx, qy) + 0.5f) / float(SUB_BINS)) /
                              float(params.map_res);
            const float4 map_dir = dir_map_dir(params.map_transform, uv);
            if (map_dir.w == 0.f) {
                continue;
            }
            const float3 wo_world = mul(transpose(to_map), map_dir.xyz);
            const float3 wo_local = frame.to_local(wo_world);
            const float pdf = path_debug_finite_or_zero(bsdf.pdf(wi_local, wo_local));
            eval_sum += path_debug_finite_or_zero(bsdf.eval(wi_local, wo_local, rng));
            pdf_sum += pdf;
            pdf_min = min(pdf_min, pdf);
            pdf_max = max(pdf_max, pdf);
            valid++;
        }
    }
    const float mean_pdf = valid > 0 ? pdf_sum / float(valid) : 0.f;
    const bool resolved = valid == SUB_BINS * SUB_BINS && pdf_max <= 8.f * pdf_min + 1e-3f;
    maps[path_debug_map_pdf_at(bin)] = asuint(resolved ? mean_pdf : -mean_pdf);
    path_debug_store_float3(maps, path_debug_map_eval_at(bin),
                            valid > 0 ? eval_sum / float(valid) : float3(0.f));
}
