import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;
import merian_shaders.colors.colormaps;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"
#include "merian-graph/nodes/path_debug/map_display.slangh"

void write_isolated() {
    const PathRecordPath p = path_record_load_path(kept, params.overlay_isolate);
    const uint vertices = min(p.vertex_count, PATH_DEBUG_MAX_SEGMENTS);
    out_state[PATH_DEBUG_OUT_ISOLATE + PATH_DEBUG_ISOLATE_VALID] = 1;
    out_state[PATH_DEBUG_OUT_ISOLATE + PATH_DEBUG_ISOLATE_PIXEL] = path_record_pack_pixel(p.pixel);
    out_state[PATH_DEBUG_OUT_ISOLATE + PATH_DEBUG_ISOLATE_SAMPLE_COUNT] =
        path_debug_pack_sample_count(p.sample_idx, vertices);
    out_state[PATH_DEBUG_OUT_ISOLATE + PATH_DEBUG_ISOLATE_LUMINANCE] = asuint(p.luminance);
    const uint src = PATH_DEBUG_KEPT_VERTICES + p.vertex_offset * PATH_RECORD_VERTEX_UINTS;
    for (uint w = 0; w < vertices * PATH_RECORD_VERTEX_UINTS; w++) {
        out_state[PATH_DEBUG_OUT_ISOLATE + PATH_DEBUG_ISOLATE_HEADER + w] = kept[src + w];
    }
}

void write_top_list() {
    uint top[PATH_DEBUG_TOP_COUNT] = {};
    float top_lum[PATH_DEBUG_TOP_COUNT] = {};
    uint filled = 0;
    for (uint slot = 0; params.want_top_list != 0 && slot < params.overlay_max_paths; slot++) {
        if (!path_debug_slot_kept(slot)) {
            continue;
        }
        const float raw = path_record_load_path(kept, slot).luminance;
        const float lum = isfinite(raw) ? raw : 1e30f;
        uint at = filled;
        while (at > 0 && top_lum[at - 1] < lum) {
            if (at < PATH_DEBUG_TOP_COUNT) {
                top[at] = top[at - 1];
                top_lum[at] = top_lum[at - 1];
            }
            at--;
        }
        if (at < PATH_DEBUG_TOP_COUNT) {
            top[at] = slot;
            top_lum[at] = lum;
            filled = min(filled + 1, PATH_DEBUG_TOP_COUNT);
        }
    }
    for (uint t = 0; t < PATH_DEBUG_TOP_COUNT; t++) {
        const uint dst = PATH_DEBUG_OUT_TOP + PATH_DEBUG_TOP_UINTS * t;
        if (t >= filled) {
            out_state[dst + PATH_DEBUG_TOP_SLOT] = PATH_DEBUG_NO_PATH;
            continue;
        }
        const uint slot = top[t];
        const PathRecordPath p = path_record_load_path(kept, slot);
        out_state[dst + PATH_DEBUG_TOP_SLOT] = slot;
        out_state[dst + PATH_DEBUG_TOP_PIXEL] = path_record_pack_pixel(p.pixel);
        out_state[dst + PATH_DEBUG_TOP_LUMINANCE] = asuint(p.luminance);
        // the deepest of the selection's light paths
        const uint2 selected = path_debug_kept_events(slot);
        out_state[dst + PATH_DEBUG_TOP_SAMPLE_SCATTER] = path_debug_pack_sample_count(
            p.sample_idx, path_debug_events_depth(kept, p, PATH_DEBUG_KEPT_VERTICES, selected));
        uint classes = 0;
        uint e = 0;
        for (int v = 0; v < path_debug_last_event(selected) && e < PATH_DEBUG_TOP_CLASSES; v++) {
            const uint meta = path_record_load_vertex_meta(kept, PATH_DEBUG_KEPT_VERTICES,
                                                           p.vertex_offset + uint(v));
            if (!path_debug_is_connection(meta)) {
                classes |= ((meta >> PATH_RECORD_LOBE_SHIFT) & PATH_RECORD_LOBE_MASK)
                           << (PATH_DEBUG_CLASS_BITS * e);
                e++;
            }
        }
        out_state[dst + PATH_DEBUG_TOP_EVENTS] = classes;
    }
}

[numthreads(1, 1, 1)]
[shader("compute")]
void main() {
    out_state[PATH_DEBUG_OUT_MATCHED] = state[PATH_DEBUG_STATE_MATCHED];
    out_state[PATH_DEBUG_OUT_GRID_CELLS] = state[PATH_DEBUG_STATE_GRID_CELLS];
    out_state[PATH_DEBUG_OUT_GRID_DROPPED] = state[PATH_DEBUG_STATE_GRID_DROPPED];
    for (uint w = 0; w < PATH_DEBUG_PICK_UINTS; w++) {
        out_state[PATH_DEBUG_OUT_PICK + w] = state[PATH_DEBUG_STATE_PICK + w];
    }
    for (uint w = 0; w < PATH_DEBUG_POINT_UINTS; w++) {
        out_state[PATH_DEBUG_OUT_POINT + w] = state[PATH_DEBUG_STATE_POINT + w];
    }

    out_state[PATH_DEBUG_OUT_PATHS] = in_records[PATH_RECORD_HDR_PATH_COUNT];
    out_state[PATH_DEBUG_OUT_VERTICES] = in_records[PATH_RECORD_HDR_VERTEX_COUNT];
    out_state[PATH_DEBUG_OUT_FRAME] = in_records[PATH_RECORD_HDR_FRAME];
    out_state[PATH_DEBUG_OUT_KEEP] = in_records[PATH_RECORD_HDR_KEEP_PROB];

    stats[PATH_DEBUG_STATS_MAX_SHARED] = 0;
    stats[PATH_DEBUG_STATS_MAX_INTEGRAND] = 0;
    stats[PATH_DEBUG_STATS_MAX_MEAN] = 0;
    stats[PATH_DEBUG_STATS_MAX_MOMENT] = 0;
    stats[PATH_DEBUG_STATS_MAX_RADIANCE] = 0;
    stats[PATH_DEBUG_STATS_EVAL_NORM] = stats[PATH_DEBUG_STATS_EVAL_INTEGRAL];
    stats[PATH_DEBUG_STATS_EVAL_INTEGRAL] = 0;
    stats[PATH_DEBUG_STATS_TV_EVAL] = 0;
    stats[PATH_DEBUG_STATS_TV_CHECK] = 0;
    stats[PATH_DEBUG_STATS_TV_RECORDS] = 0;
    stats[PATH_DEBUG_STATS_PDF_NORM] = stats[PATH_DEBUG_STATS_PDF_INTEGRAL];
    stats[PATH_DEBUG_STATS_PDF_INTEGRAL] = 0;
    stats[PATH_DEBUG_STATS_CHI2] = 0;
    stats[PATH_DEBUG_STATS_CHI2_DOF] = 0;
    stats[PATH_DEBUG_STATS_SPIKE_MASS] = 0;
    stats[PATH_DEBUG_STATS_SPIKE_BINS] = 0;
    for (uint i = 0; i < PATH_DEBUG_THETA_UINTS * PATH_DEBUG_THETA_BANDS; i++) {
        stats[PATH_DEBUG_STATS_THETA + i] = 0;
    }

    out_state[PATH_DEBUG_OUT_ISOLATE + PATH_DEBUG_ISOLATE_VALID] = 0;
    if (params.overlay_rank == PATH_DEBUG_RANK_RECENT && params.overlay_keep_update != 0) {
        kept[PATH_DEBUG_KEPT_HEAD] =
            (kept[PATH_DEBUG_KEPT_HEAD] + path_debug_recent_advance()) % params.overlay_max_paths;
    }
    uint drawn = 0;
    for (uint slot = 0; slot < params.overlay_max_paths; slot++) {
        drawn += path_debug_slot_kept(slot) ? 1 : 0;
    }
    out_state[PATH_DEBUG_OUT_DRAW_COUNT] = drawn;
    if (params.overlay_isolate < params.overlay_max_paths &&
        path_debug_slot_kept(params.overlay_isolate)) {
        write_isolated();
    }
    write_top_list();

    const PathDebugPixelMoments selected = path_debug_pixel_moments(path_debug_selected_pixel());
    const uint selected_at = PATH_DEBUG_OUT_PROBE + PATH_DEBUG_PROBE_SELECTED;
    out_state[selected_at + 0] = asuint(selected.mean);
    out_state[selected_at + 1] = asuint(selected.rel_error);
    out_state[selected_at + 2] = selected.count;
    out_state[selected_at + 3] = asuint(selected.variance);

    // out_state persists, so an empty probe has to be written
    out_state[PATH_DEBUG_OUT_PROBE + PATH_DEBUG_PROBE_KIND] = PATH_DEBUG_PROBE_NONE;
    const uint sphere_probe = state[PATH_DEBUG_STATE_SPHERE_PROBE];
    if (params.probe_channel == PATH_DEBUG_MAP_NONE) {
        if (sphere_probe != 0) {
            path_debug_write_map_probe(PATH_DEBUG_PROBE_SPHERE, params.sphere_channel,
                                       path_debug_map_cell_of(sphere_probe - 1));
        } else if (params.cursor_x != PATH_DEBUG_NO_PIXEL &&
                   params.cursor_y != PATH_DEBUG_NO_PIXEL &&
                   path_debug_source_pixel(uint2(params.cursor_x, params.cursor_y)).x !=
                       PATH_DEBUG_NO_PIXEL) {
            const PathDebugPixelMoments m = path_debug_pixel_moments(
                path_debug_source_pixel(uint2(params.cursor_x, params.cursor_y)));
            const uint values_at = PATH_DEBUG_OUT_PROBE + PATH_DEBUG_PROBE_VALUES;
            out_state[PATH_DEBUG_OUT_PROBE + PATH_DEBUG_PROBE_KIND] = PATH_DEBUG_PROBE_PIXEL;
            out_state[values_at + 0] = asuint(m.mean);
            out_state[values_at + 1] = asuint(m.rel_error);
            out_state[values_at + 2] = asuint(m.variance);
            out_state[PATH_DEBUG_OUT_PROBE + PATH_DEBUG_PROBE_COUNT] = m.count;
        }
    } else {
        path_debug_write_map_probe(PATH_DEBUG_PROBE_PANEL, params.probe_channel,
                                   path_debug_map_cell_of(params.probe_bin));
    }

    const bool hold = params.freeze != 0 && in_records[PATH_RECORD_HDR_PATH_COUNT] > 0;
    if (!hold) {
        in_records[PATH_RECORD_HDR_PATH_COUNT] = 0;
        in_records[PATH_RECORD_HDR_VERTEX_COUNT] = 0;
    }
    in_records[PATH_RECORD_HDR_ENABLE] = hold || params.capture_off != 0 ? 0 : 1;
    in_records[PATH_RECORD_HDR_PIXEL_FILTER] = params.record_pixel;
    in_records[PATH_RECORD_HDR_KEEP_PIXEL] = params.record_keep_pixel;
    in_records[PATH_RECORD_HDR_KEEP_PROB] = asuint(params.keep_prob);
}
