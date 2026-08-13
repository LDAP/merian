import merian_shaders.scene.camera;
import merian_shaders.debug.path_record;
import merian_shaders.utils.hash;

using merian;

#include "merian-graph/nodes/path_debug/layout.slangh"

// Re-renders the image from the records, keeping only the paths the filter accepts. The sums are
// normalized by the global paths-per-pixel average in compose: dividing by the realized count of
// the pixel would drop the frames where subsampling recorded nothing and bias the image dark by
// exactly the subsampling probability.
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
    if (p.vertex_count == 0 || any(p.pixel >= path_debug_dim())) {
        return;
    }

    const uint base = 4 * (p.pixel.y * params.dim_x + p.pixel.x);
    const bool matched_a = path_debug_matched(i, 0);
    const bool matched_b = path_debug_matched(i, 1);
    if (!matched_a && !matched_b) {
        return;
    }
    const uint vro = path_record_vertex_region_offset(in_records);

    float3 contrib = float3(0.f);
    for (uint k = 0; k < p.vertex_count; k++) {
        contrib += path_record_load_vertex(in_records, vro, p.vertex_offset + k).contrib;
    }
    if (all(contrib <= 0.f)) {
        return;
    }
    const uint slot_stride = 4 * params.dim_x * params.dim_y;
    if (matched_a) {
        path_debug_atomic_add_float(filtered, base + 0, contrib.r);
        path_debug_atomic_add_float(filtered, base + 1, contrib.g);
        path_debug_atomic_add_float(filtered, base + 2, contrib.b);
    }
    if (matched_b) {
        path_debug_atomic_add_float(filtered, slot_stride + base + 0, contrib.r);
        path_debug_atomic_add_float(filtered, slot_stride + base + 1, contrib.g);
        path_debug_atomic_add_float(filtered, slot_stride + base + 2, contrib.b);
    }
}
