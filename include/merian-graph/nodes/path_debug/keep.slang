import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

static const uint KEEP_THREADS = 1024;

// items: the kept slots, then the candidates above the band, then those in it
groupshared uint gs_keys[PATH_DEBUG_KEPT_SLOTS + PATH_DEBUG_CANDIDATES];
groupshared uint gs_free[PATH_DEBUG_KEPT_SLOTS];
groupshared uint gs_new[PATH_DEBUG_KEPT_SLOTS];
groupshared uint gs_free_count;
groupshared uint gs_new_count;

uint candidate_path(const uint candidate, const uint above) {
    return candidate < above
               ? state[PATH_DEBUG_STATE_CANDIDATES + candidate]
               : state[PATH_DEBUG_STATE_CANDIDATES + PATH_DEBUG_KEPT_SLOTS + candidate - above];
}

// Merges this frame's candidates into the kept top-k; a surviving path keeps its slot.
[numthreads(KEEP_THREADS, 1, 1)]
[shader("compute")]
void main(uint3 tid: SV_GroupThreadID) {
    const uint slots = params.overlay_max_paths;
    const uint k = min(state[PATH_DEBUG_STATE_KEEP_COUNT], slots);
    // brightest share ranks the frame's paths alone
    const bool per_frame = params.overlay_rank == PATH_DEBUG_RANK_BRIGHTEST_SHARE;
    const uint above = min(state[PATH_DEBUG_STATE_CANDIDATES_ABOVE], PATH_DEBUG_KEPT_SLOTS);
    const uint band = min(state[PATH_DEBUG_STATE_CANDIDATES_IN_BAND], PATH_DEBUG_BAND_CANDIDATES);
    const uint total = slots + above + band;
    if (tid.x == 0) {
        gs_free_count = 0;
        gs_new_count = 0;
    }
    for (uint item = tid.x; item < total; item += KEEP_THREADS) {
        if (item < slots) {
            gs_keys[item] = per_frame ? 0 : path_debug_kept_key_now(item);
        } else {
            const uint path_index = candidate_path(item - slots, above);
            gs_keys[item] = path_debug_top_key(path_debug_load_selected_path(path_index));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint item = tid.x; item < total; item += KEEP_THREADS) {
        const uint key = gs_keys[item];
        uint rank = 0;
        for (uint other = 0; other < total && rank < k; other++) {
            const uint other_key = gs_keys[other];
            rank += (other_key > key || (other_key == key && other < item)) ? 1 : 0;
        }
        const bool winner = key != 0 && rank < k;
        uint at;
        if (item < slots) {
            kept[PATH_DEBUG_KEPT_KEYS + item] = winner ? key : 0;
            if (!winner) {
                InterlockedAdd(gs_free_count, 1, at);
                gs_free[at] = item;
            }
        } else if (winner) {
            InterlockedAdd(gs_new_count, 1, at);
            gs_new[at] = item;
        }
    }
    AllMemoryBarrierWithGroupSync();

    for (uint n = tid.x; n < gs_new_count; n += KEEP_THREADS) {
        const uint item = gs_new[n];
        const uint path_index = candidate_path(item - slots, above);
        path_debug_keep_path(gs_free[n], path_index, path_debug_load_selected_path(path_index),
                             gs_keys[item]);
    }
}
