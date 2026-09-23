import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"
#include "merian-graph/nodes/path_debug/common.slangh"

[numthreads(PATH_DEBUG_WG, 1, 1)]
[shader("compute")]
void main(uint3 gid: SV_DispatchThreadID) {
    const uint count = path_record_path_count(in_records);
    const uint i = gid.x;
    if (i == 0) {
        path_debug_atomic_add_float(filtered, filtered_f,
                                    path_debug_filtered_paths_at(), float(count));
    }
    if (i >= count) {
        return;
    }

    const PathRecordPath p = path_record_load_path(in_records, i);
    if (p.vertex_count == 0 || any(p.pixel >= path_debug_dim())) {
        return;
    }

    const uint base = 4 * (p.pixel.y * params.dim_x + p.pixel.x);
    const bool matched_a = path_debug_matched(i, 0);
    const bool matched_b = path_debug_matched(i, 1);
    if (!matched_a && !matched_b) {
        return;
    }
    const uint vertex_region = path_record_vertex_region_offset(in_records);

    float3 contrib = float3(0.f);
    for (uint k = 0; k < p.vertex_count; k++) {
        const float3 c =
            path_record_load_vertex(in_records, vertex_region, p.vertex_offset + k).contrib;
        contrib += select(isfinite(c), c, float3(0.f));
    }
    if (all(contrib <= 0.f)) {
        return;
    }
    const uint slot_stride = path_debug_filtered_slot_stride();
    if (matched_a) {
        path_debug_atomic_add_float(filtered, filtered_f, base + 0, contrib.r);
        path_debug_atomic_add_float(filtered, filtered_f, base + 1, contrib.g);
        path_debug_atomic_add_float(filtered, filtered_f, base + 2, contrib.b);
    }
    if (matched_b) {
        path_debug_atomic_add_float(filtered, filtered_f, slot_stride + base + 0, contrib.r);
        path_debug_atomic_add_float(filtered, filtered_f, slot_stride + base + 1, contrib.g);
        path_debug_atomic_add_float(filtered, filtered_f, slot_stride + base + 2, contrib.b);
    }
}
