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
    if (p.vertex_count == 0 || any(p.pixel >= path_debug_dim()) || !path_debug_matched(i)) {
        return;
    }
    const uint vertex_region = path_record_vertex_region_offset(in_records);

    float3 contrib = float3(0.f);
    for (uint k = 0; k < p.vertex_count; k++) {
        contrib += path_debug_finite_or_zero(
            path_record_load_vertex(in_records, vertex_region, p.vertex_offset + k).contrib);
    }
    if (all(contrib <= 0.f)) {
        return;
    }
    path_debug_atomic_add_float3(filtered, filtered_f, path_debug_filtered_base(p.pixel), contrib);
}
