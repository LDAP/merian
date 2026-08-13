import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;
import merian_shaders.utils.dir_map;
import merian_shaders.colors.colorspaces;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

// Snapshots capture stats for host readback, then re-arms the record stream for the next
// iteration (the renderer only appends; all control lives here).
[numthreads(1, 1, 1)]
[shader("compute")]
void main() {
    state[PATH_DEBUG_STATE_STAT_PATHS] = in_records[PATH_RECORD_HDR_PATH_COUNT];
    state[PATH_DEBUG_STATE_STAT_VERTICES] = in_records[PATH_RECORD_HDR_VERTEX_COUNT];
    state[PATH_DEBUG_STATE_STAT_FRAME] = in_records[PATH_RECORD_HDR_FRAME];
    // the header still holds the keep probability the producer used this frame
    state[PATH_DEBUG_STATE_STAT_KEEP] = in_records[PATH_RECORD_HDR_KEEP_PROB];

    // display normalization is recomputed by map_reduce every frame
    stats[PATH_DEBUG_STATS_MAX_DENSITY] = 0;
    stats[PATH_DEBUG_STATS_MAX_CONTRIB] = 0;
    stats[PATH_DEBUG_STATS_MAX_MEAN] = 0;
    stats[PATH_DEBUG_STATS_MAX_PDF] = 0;
    stats[PATH_DEBUG_STATS_PDF_NORM] = stats[PATH_DEBUG_STATS_PDF_INTEGRAL];
    stats[PATH_DEBUG_STATS_PDF_INTEGRAL] = 0;
    stats[PATH_DEBUG_STATS_CHI2] = 0;
    stats[PATH_DEBUG_STATS_CHI2_DOF] = 0;
    stats[PATH_DEBUG_STATS_OMEGA] = 0;
    stats[PATH_DEBUG_STATS_SPIKE_MASS] = 0;
    stats[PATH_DEBUG_STATS_SPIKE_BINS] = 0;
    stats[PATH_DEBUG_STATS_TESTED] = 0;
    stats[PATH_DEBUG_STATS_AB_RELSQ] = 0;
    stats[PATH_DEBUG_STATS_AB_SIGNED] = 0;
    stats[PATH_DEBUG_STATS_AB_N] = 0;
    for (uint i = 0; i < 3 * PATH_DEBUG_THETA_BANDS; i++) {
        stats[PATH_DEBUG_STATS_THETA + i] = 0;
    }

    // dump the isolated path for the host inspector
    state[PATH_DEBUG_STATE_FOCUS] = 0;
    if (params.focus_path != PATH_DEBUG_NO_FOCUS &&
        params.focus_path < min(in_records[PATH_RECORD_HDR_PATH_COUNT],
                            path_record_path_capacity(in_records))) {
        const PathRecordPath p = path_record_load_path(in_records, params.focus_path);
        const uint vro = path_record_vertex_region_offset(in_records);
        const uint vertices = min(p.vertex_count, PATH_DEBUG_MAX_SEGMENTS);
        state[PATH_DEBUG_STATE_FOCUS + 0] = 1;
        state[PATH_DEBUG_STATE_FOCUS + 1] = path_record_pack_pixel(p.pixel);
        state[PATH_DEBUG_STATE_FOCUS + 2] = (p.sample_idx & 0xFFFFu) | (vertices << 16);
        state[PATH_DEBUG_STATE_FOCUS + 3] = asuint(p.luminance);
        for (uint v = 0; v < vertices; v++) {
            const uint src = vro + (p.vertex_offset + v) * PATH_RECORD_VERTEX_UINTS;
            const uint dst = PATH_DEBUG_STATE_FOCUS + PATH_DEBUG_FOCUS_HEADER +
                             v * PATH_RECORD_VERTEX_UINTS;
            for (uint w = 0; w < PATH_RECORD_VERTEX_UINTS; w++) {
                state[dst + w] = in_records[src + w];
            }
        }
    }

    // the brightest drawn paths, for the host inspector list. The draw list is append-ordered,
    // so a single-threaded selection pass picks the actual top by luminance; NaN orders first.
    {
        const uint drawn = min(state[PATH_DEBUG_STATE_DRAW_COUNT], params.max_draw);
        const uint vro = path_record_vertex_region_offset(in_records);
        uint top[PATH_DEBUG_TOP_COUNT];
        float top_lum[PATH_DEBUG_TOP_COUNT];
        uint filled = 0;
        for (uint d = 0; d < drawn; d++) {
            const uint candidate = state[PATH_DEBUG_STATE_DRAW_LIST + d];
            const float raw = path_record_load_path(in_records, candidate).luminance;
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
            const uint dst = PATH_DEBUG_STATE_TOP + 5 * t;
            if (t >= filled) {
                state[dst] = PATH_DEBUG_NO_FOCUS;
                continue;
            }
            const uint index = top[t];
            const PathRecordPath p = path_record_load_path(in_records, index);
            const uint scatter = path_debug_scatter_count(p, vro);
            state[dst + 0] = index;
            state[dst + 1] = path_record_pack_pixel(p.pixel);
            state[dst + 2] = asuint(p.luminance);
            state[dst + 3] = (p.sample_idx << 16) | (scatter & 0xFFFFu);
            // the first events as lobe nibbles, so the host can print the path's own expression
            uint classes = 0;
            for (uint e = 0; e < min(scatter, 8u); e++) {
                const uint meta = path_record_load_vertex_meta(in_records, vro,
                                                               p.vertex_offset + e);
                classes |= (((meta >> PATH_RECORD_LOBE_SHIFT) & PATH_RECORD_LOBE_MASK) << (4 * e));
            }
            state[dst + 4] = classes;
        }
    }

    // convergence tracking for the selected pixel
    {
        const uint m = path_debug_moments_at(path_debug_selected_pixel());
        const float sum = asfloat(moments[m + 0]);
        const float sum2 = asfloat(moments[m + 1]);
        const uint n = moments[m + 2];
        const float mean = n > 0 ? sum / float(n) : 0.f;
        const float variance = n > 1 ? max(sum2 / float(n) - mean * mean, 0.f) : 0.f;
        state[PATH_DEBUG_STATE_PROBE + 5] = asuint(mean);
        state[PATH_DEBUG_STATE_PROBE + 6] =
            asuint((n > 1 && mean > 0.f) ? sqrt(variance / float(n)) / mean : 0.f);
        state[PATH_DEBUG_STATE_PROBE + 7] = n;
    }

    // cursor probe for the hover readout
    if (params.cursor_x != PATH_DEBUG_NO_CURSOR && params.cursor_y != PATH_DEBUG_NO_CURSOR) {
        if (params.probe_view == 0) {
            const uint m = path_debug_moments_at(uint2(params.cursor_x, params.cursor_y));
            const float sum = asfloat(moments[m + 0]);
            const float sum2 = asfloat(moments[m + 1]);
            const uint n = moments[m + 2];
            const float mean = n > 0 ? sum / float(n) : 0.f;
            const float variance = n > 1 ? max(sum2 / float(n) - mean * mean, 0.f) : 0.f;
            const float rel_error = (n > 1 && mean > 0.f) ? sqrt(variance / float(n)) / mean : 0.f;
            state[PATH_DEBUG_STATE_PROBE + 0] = 1;
            state[PATH_DEBUG_STATE_PROBE + 1] = asuint(mean);
            state[PATH_DEBUG_STATE_PROBE + 2] = asuint(rel_error);
            state[PATH_DEBUG_STATE_PROBE + 3] = asuint(variance);
            state[PATH_DEBUG_STATE_PROBE + 4] = n;
        } else {
            const uint bin = params.probe_bin;
            const uint2 cell = uint2(bin % params.map_res, bin / params.map_res);
            const float solid_angle = dir_map_bin_solid_angle(params.map_transform, cell, params.map_res);
            const uint total = max(stats[PATH_DEBUG_STATS_SAMPLES], 1u);
            const uint count = maps[path_debug_map_count_at(bin)];
            const uint c = path_debug_map_contrib_at(bin);
            const float3 contrib =
                float3(asfloat(maps[c + 0]), asfloat(maps[c + 1]), asfloat(maps[c + 2]));
            const float density =
                solid_angle > 0.f ? float(count) / (float(total) * solid_angle) : 0.f;
            state[PATH_DEBUG_STATE_PROBE + 0] = 2;
            state[PATH_DEBUG_STATE_PROBE + 1] = asuint(density);
            state[PATH_DEBUG_STATE_PROBE + 2] =
                asuint(abs(asfloat(maps[path_debug_map_pdf_at(bin)])));
            state[PATH_DEBUG_STATE_PROBE + 3] =
                asuint(count > 0 ? yuv_luminance(contrib) / float(count) : 0.f);
            state[PATH_DEBUG_STATE_PROBE + 4] = count;
        }
    }

    if (params.freeze == 0) {
        in_records[PATH_RECORD_HDR_PATH_COUNT] = 0;
        in_records[PATH_RECORD_HDR_VERTEX_COUNT] = 0;
    }
    in_records[PATH_RECORD_HDR_ENABLE] = params.freeze == 0 ? 1 : 0;
    in_records[PATH_RECORD_HDR_PIXEL_FILTER] = params.pixel_filter;
    in_records[PATH_RECORD_HDR_KEEP_PROB] = asuint(params.keep_prob);
}
