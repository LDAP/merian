import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

// the group's share of the overlay histogram, the match bits and the match count, flushed once
groupshared uint gs_hist[PATH_DEBUG_HIST_BINS];
static const uint MATCH_WORDS = PATH_DEBUG_WG / 32;

groupshared uint gs_matches[MATCH_WORDS];
groupshared uint gs_matched;

void count_overlay_path(const PathRecordPath p) {
    if (!path_debug_keeps_top() || params.overlay_keep_update == 0) {
        return;
    }
    const uint key = path_debug_top_key(p);
    if (key != 0) {
        InterlockedAdd(gs_hist[path_debug_key_bin(key)], 1);
    }
}

// The first vertex past the specular transmissions at the path's start, what the pixel shows
// through clear glass; the first one if the path leaves the scene behind them.
float3 seen_through_glass(const PathRecordPath p, const uint vertex_region, const float3 first) {
    float3 seen = first;
    for (uint v = 0; v < p.vertex_count; v++) {
        const PathRecordVertex vertex =
            path_record_load_vertex(in_records, vertex_region, p.vertex_offset + v);
        if (vertex.nee) {
            continue;
        }
        if (distance(vertex.pos, camera.position) >= PATH_DEBUG_ENV_DISTANCE) {
            return first;
        }
        seen = vertex.pos;
        if (vertex.terminal ||
            vertex.lobe != (PATH_RECORD_LOBE_DELTA | PATH_RECORD_LOBE_TRANSMISSION)) {
            return seen;
        }
    }
    return seen;
}

// the per-pixel moments and the filtered image
void accumulate_pixel(const PathRecordPath p, const uint vertex_region) {
    if (params.pixel_stats_accumulate != 0 && isfinite(p.luminance)) {
        const uint base = path_debug_moments_base(p.pixel);
        path_debug_atomic_add_float(moments, moments_f, base + PATH_DEBUG_MOMENT_SUM, p.luminance);
        path_debug_atomic_add_float(moments, moments_f, base + PATH_DEBUG_MOMENT_SQUARES,
                                    p.luminance * p.luminance);
        InterlockedAdd(moments[base + PATH_DEBUG_MOMENT_COUNT], 1);
    }
    if (params.filtered_active == 0) {
        return;
    }
    float3 contrib = float3(0.f);
    for (uint k = 0; k < p.vertex_count; k++) {
        contrib += path_debug_finite_or_zero(
            path_record_load_vertex(in_records, vertex_region, p.vertex_offset + k).contrib);
    }
    if (any(contrib > 0.f)) {
        path_debug_atomic_add_float3(filtered, filtered_f, path_debug_filtered_base(p.pixel),
                                     contrib);
    }
}

void select_path(const uint i, const uint local) {
    const uint count = path_record_path_count(in_records);
    if (i >= count) {
        return;
    }

    const PathRecordPath p = path_record_load_path(in_records, i);
    if (p.vertex_count == 0 || any(p.pixel >= path_debug_dim())) {
        return;
    }
    const uint vertex_region = path_record_vertex_region_offset(in_records);

    // the exchange lets one path of the clicked pixel answer alone
    if (params.pick_x != PATH_DEBUG_NO_PIXEL &&
        all(p.pixel == uint2(params.pick_x, params.pick_y))) {
        uint claimed;
        InterlockedExchange(state[PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_RECORD_SEQ],
                            params.pick_seq, claimed);
        if (claimed != params.pick_seq) {
            const float3 first =
                path_record_load_vertex(in_records, vertex_region, p.vertex_offset).pos;
            const float3 seen = seen_through_glass(p, vertex_region, first);
            path_debug_store_float3(state, PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_FIRST, first);
            path_debug_store_float3(state, PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_SEEN, seen);
        }
    }

    if (path_debug_selects_point() && path_debug_in_selected_pixel(p)) {
        const bool at_point = path_debug_starts_at_point(p, vertex_region);
        InterlockedAdd(state[PATH_DEBUG_STATE_POINT + (at_point ? 0 : 1)], 1);
    }

    const uint scatter = path_debug_needs_scatter_count()
                             ? path_debug_scatter_count(in_records, p, vertex_region)
                             : 0;
    if (!path_debug_filter_matches(p, vertex_region, scatter) ||
        !path_debug_expression_matches(p, vertex_region, scatter)) {
        return;
    }
    // the per-pixel images leave a point out
    accumulate_pixel(p, vertex_region);
    if (!path_debug_point_matches(p, vertex_region)) {
        return;
    }
    InterlockedOr(gs_matches[local >> 5], 1u << (local & 31u));
    InterlockedAdd(gs_matched, 1);
    count_overlay_path(p);
}

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID, uint3 tid: SV_GroupThreadID) {
    if (tid.x < PATH_DEBUG_HIST_BINS) {
        gs_hist[tid.x] = 0;
    }
    if (tid.x < MATCH_WORDS) {
        gs_matches[tid.x] = 0;
    }
    if (tid.x == 0) {
        gs_matched = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (gid.x == 0 && params.filtered_active != 0) {
        path_debug_atomic_add_float(filtered, filtered_f, path_debug_filtered_paths_at(),
                                    float(path_record_path_count(in_records)));
    }
    select_path(gid.x, tid.x);
    GroupMemoryBarrierWithGroupSync();
    if (tid.x < PATH_DEBUG_HIST_BINS && gs_hist[tid.x] != 0) {
        InterlockedAdd(state[PATH_DEBUG_STATE_HIST + tid.x], gs_hist[tid.x]);
    }
    // the group starts at a multiple of PATH_DEBUG_WG, so its bits fill whole words
    if (tid.x < MATCH_WORDS && gs_matches[tid.x] != 0) {
        InterlockedOr(matches[(gid.x - tid.x) / 32 + tid.x], gs_matches[tid.x]);
    }
    if (tid.x == 0 && gs_matched != 0) {
        InterlockedAdd(state[PATH_DEBUG_STATE_MATCHED], gs_matched);
    }
}
