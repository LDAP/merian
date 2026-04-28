[vk::constant_id(0)]
const int workgroup_size_x = 64;

import merian_shaders.scene.scene_data;

using merian;

[vk::binding(0)]
StructuredBuffer<PackedPrevVertexData, ScalarDataLayout> src_vertices;
[vk::binding(1)]
RWStructuredBuffer<PackedPrevVertexData, ScalarDataLayout> dst_vertices;

struct PushConstant {
    float4x4 transform;
    uint vertex_count;
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

[numthreads(workgroup_size_x, 1, 1)]
[shader("compute")]
void main(uint id: SV_DispatchThreadID) {
    if (id >= pc.vertex_count)
        return;

    PackedPrevVertexData out_v;
    out_v.position = mul(pc.transform, float4(src_vertices[id].position, 1.0)).xyz;
    dst_vertices[id] = out_v;
}
