[vk::constant_id(0)]
const int workgroup_size_x = 64;

import merian_shaders.utils.encoding;
import merian_shaders.scene.scene_data;

using merian;

[vk::binding(0)]
StructuredBuffer<PackedVertexData, ScalarDataLayout> src_vertices;
[vk::binding(1)]
RWStructuredBuffer<PackedVertexData, ScalarDataLayout> dst_vertices;

struct PushConstant {
    float4x4 transform;
    float4x4 inverse_transposed;
    uint vertex_count;
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

[numthreads(workgroup_size_x, 1, 1)]
[shader("compute")]
void main(uint id: SV_DispatchThreadID) {
    if (id >= pc.vertex_count)
        return;

    const PackedVertexData v = src_vertices[id];

    const float3 normal = decode_normal(v.encoded_normal);
    const float3 tangent_dir = decode_normal(v.encoded_tangent & ~1u);
    const uint tangent_sign_bit = v.encoded_tangent & 1u;

    PackedVertexData out_v;
    out_v.position = mul(pc.transform, float4(v.position, 1.0)).xyz;
    out_v.encoded_normal =
        encode_normal(normalize(mul(pc.inverse_transposed, float4(normal, 0.0)).xyz));
    out_v.uv = v.uv;
    const float3 tangent_world =
        normalize(mul(pc.inverse_transposed, float4(tangent_dir, 0.0)).xyz);
    out_v.encoded_tangent = (encode_normal(tangent_world) & ~1u) | tangent_sign_bit;

    dst_vertices[id] = out_v;
}
