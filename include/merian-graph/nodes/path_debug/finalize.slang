import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

void write_isolated(PathRecordStream records, const uint vertex_region) {
    const PathRecordPath p = path_record_load_path(records, params.overlay_isolate);
    const uint vertices = min(p.vertex_count, PATH_DEBUG_MAX_SEGMENTS);
    out_state[PATH_DEBUG_OUT_ISOLATE + 0] = 1;
    out_state[PATH_DEBUG_OUT_ISOLATE + 1] = path_record_pack_pixel(p.pixel);
    out_state[PATH_DEBUG_OUT_ISOLATE + 2] = (p.sample_idx & 0xFFFFu) | (vertices << 16);
    out_state[PATH_DEBUG_OUT_ISOLATE + 3] = asuint(p.luminance);
    for (uint v = 0; v < vertices; v++) {
        const uint src = vertex_region + (p.vertex_offset + v) * PATH_RECORD_VERTEX_UINTS;
        const uint dst =
            PATH_DEBUG_OUT_ISOLATE + PATH_DEBUG_ISOLATE_HEADER + v * PATH_RECORD_VERTEX_UINTS;
        for (uint w = 0; w < PATH_RECORD_VERTEX_UINTS; w++) {
            out_state[dst + w] = records[src + w];
        }
    }
}

void write_top_list(PathRecordStream records,
                    const uint vertex_region,
                    const bool slots,
                    const uint candidates) {
    if (params.want_top_list == 0) {
        for (uint t = 0; t < PATH_DEBUG_TOP_COUNT; t++) {
            out_state[PATH_DEBUG_OUT_TOP + PATH_DEBUG_TOP_UINTS * t] = PATH_DEBUG_NO_ISOLATE;
        }
        return;
    }
    uint top[PATH_DEBUG_TOP_COUNT] = {};
    float top_lum[PATH_DEBUG_TOP_COUNT] = {};
    uint filled = 0;
    for (uint d = 0; d < candidates; d++) {
        if (slots && kept[PATH_DEBUG_KEPT_KEYS + d] == 0) {
            continue;
        }
        const uint candidate = slots ? d : state[PATH_DEBUG_STATE_DRAW_LIST + d];
        const float raw = asfloat(records[PATH_RECORD_HEADER_UINTS + candidate * PATH_RECORD_UINTS +
                                          PATH_RECORD_PATH_LUMINANCE]);
        const float lum = (isnan(raw) || isinf(raw)) ? 1e30f : raw;
        uint at = filled;
        while (at > 0 && top_lum[at - 1] < lum) {
            if (at < PATH_DEBUG_TOP_COUNT) {
                top[at] = top[at - 1];
                top_lum[at] = top_lum[at - 1];
            }
            at--;
        }
        if (at < PATH_DEBUG_TOP_COUNT) {
            top[at] = candidate;
            top_lum[at] = lum;
            filled = min(filled + 1, PATH_DEBUG_TOP_COUNT);
        }
    }
    for (uint t = 0; t < PATH_DEBUG_TOP_COUNT; t++) {
        const uint dst = PATH_DEBUG_OUT_TOP + PATH_DEBUG_TOP_UINTS * t;
        if (t >= filled) {
            out_state[dst] = PATH_DEBUG_NO_ISOLATE;
            continue;
        }
        const uint index = top[t];
        const PathRecordPath p = path_record_load_path(records, index);
        const uint scatter = path_debug_scatter_count(records, p, vertex_region);
        out_state[dst + 0] = index;
        out_state[dst + 1] = path_record_pack_pixel(p.pixel);
        out_state[dst + 2] = asuint(p.luminance);
        out_state[dst + 3] = (p.sample_idx & 0xFFFFu) | (scatter << 16);
        uint classes = 0;
        uint e = 0;
        for (uint v = 0; v < p.vertex_count && e < 8u; v++) {
            const uint meta =
                path_record_load_vertex_meta(records, vertex_region, p.vertex_offset + v);
            if (path_debug_is_event(meta)) {
                classes |= ((meta >> PATH_RECORD_LOBE_SHIFT) & PATH_RECORD_LOBE_MASK) << (4 * e);
                e++;
            }
        }
        out_state[dst + 4] = classes;
    }
}

[numthreads(1, 1, 1)]
[shader("compute")]
void main() {
    out_state[PATH_DEBUG_OUT_DRAW_COUNT] = state[PATH_DEBUG_STATE_DRAW_COUNT];
    out_state[PATH_DEBUG_OUT_MATCHED] = state[PATH_DEBUG_STATE_MATCHED];
    out_state[PATH_DEBUG_OUT_GRID_CELLS] = state[PATH_DEBUG_STATE_GRID_CELLS];
    out_state[PATH_DEBUG_OUT_GRID_DROPPED] = state[PATH_DEBUG_STATE_GRID_DROPPED];
    for (uint w = 0; w < PATH_DEBUG_PICK_UINTS; w++) {
        out_state[PATH_DEBUG_OUT_PICK + w] = state[PATH_DEBUG_STATE_PICK + w];
    }

    out_state[PATH_DEBUG_OUT_PATHS] = in_records[PATH_RECORD_HDR_PATH_COUNT];
    out_state[PATH_DEBUG_OUT_VERTICES] = in_records[PATH_RECORD_HDR_VERTEX_COUNT];
    out_state[PATH_DEBUG_OUT_FRAME] = in_records[PATH_RECORD_HDR_FRAME];
    out_state[PATH_DEBUG_OUT_KEEP] = in_records[PATH_RECORD_HDR_KEEP_PROB];

    stats[PATH_DEBUG_STATS_MAX_DENSITY] = 0;
    stats[PATH_DEBUG_STATS_MAX_CONTRIB] = 0;
    stats[PATH_DEBUG_STATS_MAX_MEAN] = 0;
    stats[PATH_DEBUG_STATS_MAX_PDF] = 0;
    stats[PATH_DEBUG_STATS_MAX_CHECK] = 0;
    stats[PATH_DEBUG_STATS_MAX_EVAL] = 0;
    stats[PATH_DEBUG_STATS_EVAL_NORM] = stats[PATH_DEBUG_STATS_EVAL_INTEGRAL];
    stats[PATH_DEBUG_STATS_EVAL_INTEGRAL] = 0;
    stats[PATH_DEBUG_STATS_TV_EVAL] = 0;
    stats[PATH_DEBUG_STATS_TV_CHECK] = 0;
    stats[PATH_DEBUG_STATS_TV_RECORDS] = 0;
    stats[PATH_DEBUG_STATS_PDF_NORM] = stats[PATH_DEBUG_STATS_PDF_INTEGRAL];
    stats[PATH_DEBUG_STATS_PDF_INTEGRAL] = 0;
    stats[PATH_DEBUG_STATS_CHI2] = 0;
    stats[PATH_DEBUG_STATS_CHI2_DOF] = 0;
    stats[PATH_DEBUG_STATS_OMEGA] = 0;
    stats[PATH_DEBUG_STATS_SPIKE_MASS] = 0;
    stats[PATH_DEBUG_STATS_SPIKE_BINS] = 0;
    stats[PATH_DEBUG_STATS_TESTED] = 0;
    for (uint i = 0; i < 3 * PATH_DEBUG_THETA_BANDS; i++) {
        stats[PATH_DEBUG_STATS_THETA + i] = 0;
    }

    out_state[PATH_DEBUG_OUT_ISOLATE] = 0;
    if (params.overlay_rank == PATH_DEBUG_RANK_RECENT && params.overlay_keep_update != 0) {
        kept[PATH_DEBUG_KEPT_HEAD] =
            (kept[PATH_DEBUG_KEPT_HEAD] + path_debug_recent_advance()) % params.overlay_max_paths;
    }
    if (path_debug_overlay_kept()) {
        uint filled = 0;
        for (uint slot = 0; slot < params.overlay_max_paths; slot++) {
            filled += kept[PATH_DEBUG_KEPT_KEYS + slot] != 0 ? 1 : 0;
        }
        out_state[PATH_DEBUG_OUT_DRAW_COUNT] = filled;
        if (params.overlay_isolate < params.overlay_max_paths &&
            kept[PATH_DEBUG_KEPT_KEYS + params.overlay_isolate] != 0) {
            write_isolated(kept, PATH_DEBUG_KEPT_VERTICES);
        }
        write_top_list(kept, PATH_DEBUG_KEPT_VERTICES, true, params.overlay_max_paths);
    } else {
        if (params.overlay_isolate < path_record_path_count(in_records)) {
            write_isolated(in_records, path_record_vertex_region_offset(in_records));
        }
        write_top_list(in_records, path_record_vertex_region_offset(in_records), false,
                       min(state[PATH_DEBUG_STATE_DRAW_COUNT], params.overlay_max_paths));
    }

    const PixelMoments picked = path_debug_pixel_moments(path_debug_picked_pixel());
    out_state[PATH_DEBUG_OUT_PROBE + 5] = asuint(picked.mean);
    out_state[PATH_DEBUG_OUT_PROBE + 6] = asuint(picked.rel_error);
    out_state[PATH_DEBUG_OUT_PROBE + 7] = picked.count;
    out_state[PATH_DEBUG_OUT_PROBE + 8] = asuint(picked.variance);

    // out_state persists, so an empty probe has to be written
    out_state[PATH_DEBUG_OUT_PROBE + 0] = PATH_DEBUG_PROBE_NONE;
    if (params.cursor_x != PATH_DEBUG_NO_PIXEL && params.cursor_y != PATH_DEBUG_NO_PIXEL) {
        if (params.probe_channel == PATH_DEBUG_MAP_NONE) {
            const PixelMoments m =
                path_debug_pixel_moments(uint2(params.cursor_x, params.cursor_y));
            out_state[PATH_DEBUG_OUT_PROBE + 0] = PATH_DEBUG_PROBE_PIXEL;
            out_state[PATH_DEBUG_OUT_PROBE + 1] = asuint(m.mean);
            out_state[PATH_DEBUG_OUT_PROBE + 2] = asuint(m.rel_error);
            out_state[PATH_DEBUG_OUT_PROBE + 3] = asuint(m.variance);
            out_state[PATH_DEBUG_OUT_PROBE + 4] = m.count;
        } else {
            const uint bin = params.probe_bin;
            const uint2 cell = uint2(bin % params.map_res, bin / params.map_res);
            const float solid_angle =
                dir_map_bin_solid_angle(params.map_transform, cell, params.map_res);
            const bool check = params.probe_channel == PATH_DEBUG_MAP_CHECK;
            const uint total =
                max(stats[check ? PATH_DEBUG_STATS_CHECK_SAMPLES : PATH_DEBUG_STATS_SAMPLES], 1u);
            const uint count =
                maps[check ? path_debug_map_check_at(bin) : path_debug_map_count_at(bin)];
            const uint c = path_debug_map_contrib_at(bin);
            const float3 contrib =
                float3(asfloat(maps[c + 0]), asfloat(maps[c + 1]), asfloat(maps[c + 2]));
            const float density =
                solid_angle > 0.f ? float(count) / (float(total) * solid_angle) : 0.f;
            out_state[PATH_DEBUG_OUT_PROBE + 0] = PATH_DEBUG_PROBE_PANEL;
            out_state[PATH_DEBUG_OUT_PROBE + 1] = asuint(density);
            out_state[PATH_DEBUG_OUT_PROBE + 2] =
                asuint(abs(asfloat(maps[path_debug_map_pdf_at(bin)])));
            out_state[PATH_DEBUG_OUT_PROBE + 3] =
                asuint(count > 0 ? yuv_luminance(contrib) / float(count) : 0.f);
            out_state[PATH_DEBUG_OUT_PROBE + 4] = count;
        }
    }

    const bool hold = params.freeze != 0 && in_records[PATH_RECORD_HDR_PATH_COUNT] > 0;
    if (!hold) {
        in_records[PATH_RECORD_HDR_PATH_COUNT] = 0;
        in_records[PATH_RECORD_HDR_VERTEX_COUNT] = 0;
    }
    in_records[PATH_RECORD_HDR_ENABLE] = hold ? 0 : 1;
    in_records[PATH_RECORD_HDR_PIXEL_FILTER] = params.record_pixel;
    in_records[PATH_RECORD_HDR_KEEP_PROB] = asuint(params.keep_prob);
}
