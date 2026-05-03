[vk::constant_id(0)]
const int workgroup_size_x = 256;

import merian_shaders.scene.scene_data;

using merian;

[vk::binding(0)]
StructuredBuffer<TransformPrevVertexJob, ScalarDataLayout> jobs;

[numthreads(workgroup_size_x, 1, 1)]
[shader("compute")]
void main(uint thread_id: SV_GroupThreadID, uint2 wg_id: SV_GroupID) {
    const TransformPrevVertexJob job = jobs[wg_id.y];
    const uint i = wg_id.x * workgroup_size_x + thread_id;
    if (i >= job.vertex_count)
        return;

    PackedPrevVertexData p = job.src[i];
    p.position = mul(job.prev_transform, float4(p.position, 1.0)).xyz;
    job.dst[i] = p;
}
