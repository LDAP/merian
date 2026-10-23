import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.colors.colorspaces;

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
PathRecordVertex seen_through_glass(const PathRecordPath p, const uint vertex_region) {
    const PathRecordVertex first =
        path_record_load_vertex(in_records, vertex_region, p.vertex_offset);
    PathRecordVertex seen = first;
    for (uint v = 0; v < p.vertex_count; v++) {
        const PathRecordVertex vertex =
            path_record_load_vertex(in_records, vertex_region, p.vertex_offset + v);
        if (vertex.nee) {
            continue;
        }
        if (distance(vertex.pos, camera.position) >= PATH_DEBUG_ENV_DISTANCE) {
            return first;
        }
        seen = vertex;
        if (vertex.terminal ||
            vertex.lobe != (PATH_RECORD_LOBE_DELTA | PATH_RECORD_LOBE_TRANSMISSION)) {
            return seen;
        }
    }
    return seen;
}

// every path at its pixel: the moments of what the selection adds, and the filtered image
void accumulate_pixel(const PathRecordPath p, const PathDebugEvents selected) {
    const float luminance = max(yuv_luminance(selected.contrib), 0.f);
    const bool held = any(selected.mask != 0);
    if (params.pixel_stats_accumulate != 0 && isfinite(luminance)) {
        const uint base = path_debug_moments_base(p.pixel);
        InterlockedAdd(moments[base + PATH_DEBUG_MOMENT_COUNT], 1);
        if (luminance > 0.f) {
            path_debug_atomic_add_float(moments, moments_f, base + PATH_DEBUG_MOMENT_SUM,
                                        luminance);
            path_debug_atomic_add_float(moments, moments_f, base + PATH_DEBUG_MOMENT_SQUARES,
                                        luminance * luminance);
            InterlockedMax(moments[base + PATH_DEBUG_MOMENT_MAX], asuint(luminance));
        } else {
            InterlockedAdd(moments[base + PATH_DEBUG_MOMENT_ZEROS], 1);
        }
    }
    if (params.image_accumulate == 0) {
        return;
    }
    const uint base = path_debug_filtered_base(p.pixel);
    InterlockedAdd(filtered[base + PATH_DEBUG_FILTERED_COUNT], 1);
    const float3 contrib = path_debug_finite_or_zero(selected.contrib);
    if (held && any(contrib > 0.f)) {
        path_debug_atomic_add_float3(filtered, filtered_f, base, contrib);
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
            const PathRecordVertex seen = seen_through_glass(p, vertex_region);
            path_debug_store_float3(state, PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_FIRST, first);
            path_debug_store_float3(state, PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_SEEN, seen.pos);
            state[PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_SEEN_SURFACE] = seen.surface.x;
            state[PATH_DEBUG_STATE_PICK + PATH_DEBUG_PICK_SEEN_SURFACE + 1] = seen.surface.y;
        }
    }

    if (path_debug_selects_point() && path_debug_in_selected_pixel(p)) {
        const bool at_point = path_debug_starts_at_point(p, vertex_region);
        InterlockedAdd(state[PATH_DEBUG_STATE_POINT + (at_point ? 0 : 1)], 1);
    }

    const PathDebugEvents selected = path_debug_match_events(p, vertex_region);
    // the per-pixel images leave a point out
    accumulate_pixel(p, selected);
    if ((all(selected.mask == 0) && !path_debug_selection_open()) ||
        !path_debug_point_matches(p, vertex_region)) {
        return;
    }
    PathRecordPath held = p;
    held.luminance = yuv_luminance(selected.contrib);
    events[i * PATH_DEBUG_EVENT_UINTS] = selected.mask.x;
    events[i * PATH_DEBUG_EVENT_UINTS + 1] = selected.mask.y;
    events[i * PATH_DEBUG_EVENT_UINTS + 2] = asuint(held.luminance);
    InterlockedOr(gs_matches[local >> 5], 1u << (local & 31u));
    InterlockedAdd(gs_matched, 1);
    count_overlay_path(held);
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
