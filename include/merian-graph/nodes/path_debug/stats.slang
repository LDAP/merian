import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;
import merian_shaders.gbuffer;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

ParameterBlock<GBuffer> gbuffer;

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint count = path_record_path_count(in_records);
    const uint phase =
        params.stats_stride > 1 ? pcg(in_records[PATH_RECORD_HDR_FRAME]) % params.stats_stride : 0;
    const uint i = gid.x * params.stats_stride + phase;
    if (i >= count) {
        return;
    }

    const PathRecordPath p = path_record_load_path(in_records, i);
    if (p.vertex_count == 0) {
        return;
    }
    if (!path_debug_in_source(i, params.map_source)) {
        return;
    }
    const uint vertex_region = path_record_vertex_region_offset(in_records);
    const uint scatter = path_debug_scatter_count(in_records, p, vertex_region);

    InterlockedAdd(stats[PATH_DEBUG_STATS_COUNT], 1);
    InterlockedAdd(stats[PATH_DEBUG_STATS_LUM +
                         path_debug_log_bin(p.luminance, PATH_DEBUG_HIST_MIN_LOG,
                                            PATH_DEBUG_HIST_MAX_LOG, PATH_DEBUG_HIST_SMALL_BINS)],
                   1);
    const uint len_bin = min(scatter, PATH_DEBUG_LENGTH_BINS - 1);
    InterlockedAdd(stats[PATH_DEBUG_STATS_BOUNCE + len_bin], 1);
    if (!isnan(p.luminance) && !isinf(p.luminance)) {
        path_debug_atomic_add_float(stats, stats_f, PATH_DEBUG_STATS_LEN_CONTRIB + len_bin,
                                    p.luminance);
    }

    const bool local_frame =
        merian_path_debug_gbuffer && params.map_frame == PATH_DEBUG_FRAME_LOCAL;
    const float3x3 to_map = dir_map_frame(
        local_frame ? gbuffer.get_normal(path_debug_picked_pixel()) : float3(0, 0, 1), local_frame);

    uint chain = 0;
    for (uint v = 0; v < p.vertex_count; v++) {
        chain += path_debug_is_connection(
                     path_record_load_vertex_meta(in_records, vertex_region, p.vertex_offset + v))
                     ? 0
                     : 1;
    }

    uint samples_acc = 0;
    uint draws_acc = 0;
    uint seen = 0;
    float3 downstream = float3(0.f);
    float3 nee_at_parent = float3(0.f);
    for (int k = int(p.vertex_count) - 1; k >= 0; k--) {
        const PathRecordVertex v =
            path_record_load_vertex(in_records, vertex_region, p.vertex_offset + k);
        const float3 contrib = select(isfinite(v.contrib), v.contrib, float3(0.f));
        float3 transported = contrib;
        if (v.nee) {
            nee_at_parent += contrib;
        } else {
            transported = downstream;
            downstream += contrib + nee_at_parent;
            nee_at_parent = float3(0.f);
            chain--;
        }
        const uint index = v.nee ? chain - 1 : chain;
        if (params.map_bounce != PATH_DEBUG_ALL_BOUNCES && index != params.map_bounce) {
            continue;
        }
        if (v.terminal || v.nee != (params.map_draws == PATH_DEBUG_DRAWS_LIGHT)) {
            continue;
        }
        if (v.nee) {
            seen |= PATH_DEBUG_SEEN_CONNECTION | (v.reused ? PATH_DEBUG_SEEN_REUSED : 0);
            if (path_debug_is_empty_draw(v)) {
                draws_acc++;
                continue;
            }
        }
        if (v.resampled) {
            seen |= PATH_DEBUG_SEEN_RESAMPLED;
        }
        const bool continuous =
            (v.lobe & PATH_DEBUG_CLS_DELTA) == 0 && v.pdf > 0.f && !v.resampled;
        if (continuous) {
            draws_acc++;
        }

        const float3 uvw = dir_map_uv(params.map_transform, mul(to_map, v.wo));
        if (uvw.z != 0.f) {
            const uint2 cell = min(uint2(uvw.xy * params.map_res), params.map_res - 1);
            const uint bin = cell.y * params.map_res + cell.x;
            InterlockedAdd(maps[path_debug_map_count_at(bin)], 1);
            const uint contrib = path_debug_map_contrib_at(bin);
            path_debug_atomic_add_float(maps, maps_f, contrib + 0, transported.r);
            path_debug_atomic_add_float(maps, maps_f, contrib + 1, transported.g);
            path_debug_atomic_add_float(maps, maps_f, contrib + 2, transported.b);
            samples_acc++;
            if (continuous) {
                const float recip = min(1.f / v.pdf, 1e12f);
                path_debug_atomic_add_float(maps, maps_f, path_debug_map_recip_at(bin), recip);
                path_debug_atomic_add_float(maps, maps_f, path_debug_map_recip2_at(bin),
                                            recip * recip);
            }
        }

        InterlockedAdd(
            stats[PATH_DEBUG_STATS_PDF + path_debug_log_bin(v.pdf, PATH_DEBUG_PDF_MIN_LOG,
                                                            PATH_DEBUG_PDF_MAX_LOG,
                                                            PATH_DEBUG_HIST_SMALL_BINS)],
            1);
    }
    if (samples_acc != 0) {
        InterlockedAdd(stats[PATH_DEBUG_STATS_SAMPLES], samples_acc);
    }
    if (draws_acc != 0) {
        InterlockedAdd(stats[PATH_DEBUG_STATS_DRAWS], draws_acc);
    }
    if (seen != 0) {
        InterlockedOr(stats[PATH_DEBUG_STATS_SEEN], seen);
    }
}
