[vk::constant_id(0)]
const int workgroup_size_x = 256;

import merian_shaders.utils.encoding;
import merian_shaders.scene.scene_data;

using merian;

[vk::binding(0)]
StructuredBuffer<TransformVertexJob, ScalarDataLayout> jobs;

[numthreads(workgroup_size_x, 1, 1)]
[shader("compute")]
void main(uint thread_id: SV_GroupThreadID, uint2 wg_id: SV_GroupID) {
    const TransformVertexJob job = jobs[wg_id.y];
    const uint i = wg_id.x * workgroup_size_x + thread_id;
    if (i >= job.vertex_count)
        return;

    const PackedVertexData v_in = job.src[i];

    if (job.prev_dst != nullptr) {
        PackedPrevVertexData p;
        p.position = mul(job.prev_transform, float4(v_in.position, 1.0)).xyz;
        job.prev_dst[i] = p;
    }

    const float3 normal = decode_normal(v_in.encoded_normal);
    const float3 t_dir = decode_normal(v_in.encoded_tangent & ~1u);
    const uint t_sign = v_in.encoded_tangent & 1u;

    PackedVertexData v_out;
    v_out.position = mul(job.transform, float4(v_in.position, 1.0)).xyz;
    v_out.encoded_normal =
        encode_normal(normalize(mul(job.inverse_transposed, float4(normal, 0.0)).xyz));
    v_out.uv = v_in.uv;
    const float3 t_world =
        normalize(mul(job.inverse_transposed, float4(t_dir, 0.0)).xyz);
    v_out.encoded_tangent = (encode_normal(t_world) & ~1u) | t_sign;

    job.dst[i] = v_out;
}
