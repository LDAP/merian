import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.utils.encoding;
import merian_shaders.utils.frame;
import merian_shaders.colors.colorspaces;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

ParameterBlock<GBuffer> gbuffer;

uint log_bin(const float value, const float min_log, const float max_log, const uint bins) {
    if (!(value > 0.f)) {
        return 0;
    }
    const float normalized = (log2(value) - min_log) / (max_log - min_log);
    return clamp(uint(normalized * (bins - 2) + 1), 1, bins - 1);
}

// Accumulates per-pixel (or image-wide) sampling statistics: the directional maps (sample count
// and transported contribution) plus luminance/pdf/bounce histograms.
[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint count =
        min(in_records[PATH_RECORD_HDR_PATH_COUNT], path_record_path_capacity(in_records));
    const uint i = gid.x;
    if (i >= count) {
        return;
    }

    const PathRecordPath p = path_record_load_path(in_records, i);
    if (p.vertex_count == 0) {
        return;
    }
    if (params.map_scope == 0 && any(p.pixel != path_debug_selected_pixel())) {
        return;
    }

    if (!path_debug_matched(i, 0)) {
        return;
    }
    const uint vro = path_record_vertex_region_offset(in_records);
    const uint scatter = path_debug_scatter_count(p, vro);

    InterlockedAdd(stats[PATH_DEBUG_STATS_COUNT], 1);
    InterlockedAdd(stats[PATH_DEBUG_STATS_LUM + log_bin(p.luminance, PATH_DEBUG_HIST_MIN_LOG,
                                                        PATH_DEBUG_HIST_MAX_LOG, 64)],
                   1);
    const uint len_bin = min(scatter, 31u);
    InterlockedAdd(stats[PATH_DEBUG_STATS_BOUNCE + len_bin], 1);
    path_debug_atomic_add_float(stats, PATH_DEBUG_STATS_LEN_CONTRIB + len_bin, p.luminance);

    // binning frame; the local shading frame comes from the gbuffer normal so it matches bsdf_map
    float3x3 to_map = dir_map_frame(float3(0, 0, 1), false);
    if (merian_path_debug_gbuffer && params.map_frame == 1) {
        to_map = dir_map_frame(gbuffer.get_normal(path_debug_selected_pixel()), true);
    }

    // radiance transported through the direction sampled at each vertex, accumulated backwards
    float3 downstream = float3(0.f);
    for (int k = int(p.vertex_count) - 1; k >= 0; k--) {
        const PathRecordVertex v = path_record_load_vertex(in_records, vro, p.vertex_offset + k);
        const float3 transported = downstream;
        downstream += v.contrib;
        if (params.map_bounce != PATH_DEBUG_ALL_BOUNCES && uint(k) != params.map_bounce) {
            continue;
        }
        if (v.terminal) {
            continue;
        }
        // delta events carry a probability mass, not a density: they stay out of the
        // reciprocal-pdf identity on both sides
        const bool continuous = (v.lobe & PATH_DEBUG_CLS_DELTA) == 0 && v.pdf > 0.f;
        if (continuous) {
            InterlockedAdd(stats[PATH_DEBUG_STATS_DRAWS], 1);
        }

        const float3 uvw = dir_map_uv(params.map_transform, mul(to_map, v.wo));
        if (uvw.z != 0.f) {
            const uint2 cell = min(uint2(uvw.xy * params.map_res), params.map_res - 1);
            const uint bin = cell.y * params.map_res + cell.x;
            InterlockedAdd(maps[path_debug_map_count_at(bin)], 1);
            const uint contrib = path_debug_map_contrib_at(bin);
            path_debug_atomic_add_float(maps, contrib + 0, transported.r);
            path_debug_atomic_add_float(maps, contrib + 1, transported.g);
            path_debug_atomic_add_float(maps, contrib + 2, transported.b);
            InterlockedAdd(stats[PATH_DEBUG_STATS_SAMPLES], 1);
            if (continuous) {
                const float recip = min(1.f / v.pdf, 1e12f);
                path_debug_atomic_add_float(maps, path_debug_map_recip_at(bin), recip);
                path_debug_atomic_add_float(maps, path_debug_map_recip2_at(bin), recip * recip);
            }
        }

        InterlockedAdd(stats[PATH_DEBUG_STATS_PDF + log_bin(v.pdf, PATH_DEBUG_PDF_MIN_LOG,
                                                            PATH_DEBUG_PDF_MAX_LOG, 64)],
                       1);
    }
}
