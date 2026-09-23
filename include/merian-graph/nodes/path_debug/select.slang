import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

void count_overlay_path(const uint path_index, const float luminance) {
    InterlockedAdd(state[PATH_DEBUG_STATE_MATCHED], 1);
    path_debug_resample_offer(path_index, luminance);
    if (params.overlay_rank != PATH_DEBUG_RANK_ANY) {
        InterlockedAdd(state[PATH_DEBUG_STATE_HIST + path_debug_lum_bin(luminance)], 1);
    }
}

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint count = path_record_path_count(in_records);
    const uint i = gid.x;
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
        InterlockedExchange(state[PATH_DEBUG_STATE_PICK], params.pick_seq, claimed);
        if (claimed != params.pick_seq) {
            const float3 pos =
                path_record_load_vertex(in_records, vertex_region, p.vertex_offset).pos;
            state[PATH_DEBUG_STATE_PICK + 1] = asuint(pos.x);
            state[PATH_DEBUG_STATE_PICK + 2] = asuint(pos.y);
            state[PATH_DEBUG_STATE_PICK + 3] = asuint(pos.z);
            state[PATH_DEBUG_STATE_PICK + 4] = asuint(
                max(path_debug_pixel_footprint(p.pixel, distance(camera.position, pos)), 1e-6f));
        }
    }

    if (params.overlay_source == PATH_DEBUG_SOURCE_ALL) {
        count_overlay_path(i, p.luminance);
    }

    const bool finite = !isnan(p.luminance) && !isinf(p.luminance);
    const uint moments_base = path_debug_moments_base(p.pixel);
    if (params.pixel_stats_accumulate != 0 && finite) {
        path_debug_atomic_add_float(moments, moments_f, moments_base + 0, p.luminance);
        path_debug_atomic_add_float(moments, moments_f, moments_base + 1,
                                    p.luminance * p.luminance);
        InterlockedAdd(moments[moments_base + 2], 1);
    }

    const uint scatter = path_debug_scatter_count(in_records, p, vertex_region);
    if (!path_debug_selection_matches(p, vertex_region, scatter)) {
        return;
    }
    const bool in_a = path_debug_expression_matches(0, p, vertex_region, scatter);
    const bool in_b = path_debug_expression_matches(1, p, vertex_region, scatter);
    if (in_a) {
        path_debug_set_match(i, 0);
    }
    if (in_b) {
        path_debug_set_match(i, 1);
    }
    if (!in_a && !in_b) {
        return;
    }

    if (params.pixel_stats_accumulate != 0 && finite &&
        params.pixel_stats_source != PATH_DEBUG_SOURCE_ALL &&
        (params.pixel_stats_source == PATH_DEBUG_SOURCE_A ? in_a : in_b)) {
        path_debug_atomic_add_float(moments, moments_f, moments_base + 3, p.luminance);
        path_debug_atomic_add_float(moments, moments_f, moments_base + 4,
                                    p.luminance * p.luminance);
        InterlockedAdd(moments[moments_base + 5], 1);
    }

    if (params.overlay_source != PATH_DEBUG_SOURCE_ALL &&
        path_debug_in_source(i, params.overlay_source)) {
        count_overlay_path(i, p.luminance);
    }
}
