import merian_shaders.colors.colorspaces;
import merian_shaders.utils.textures;

struct PushConstant {
    float firefly_percentile_lower;
    float firefly_percentile_upper;
    float adaptive_alpha_percentile_lower;
    float adaptive_alpha_percentile_upper;
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

// graph io
Sampler2D<float4> in_src;

// percentile pass result, bound by the node
RWTexture2D<float4> quartiles;

groupshared float luminance[workgroup_size_x * workgroup_size_y];
groupshared float percentile[4];

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(uint3 gl_WorkGroupID: SV_GroupID,
          uint3 gl_GlobalInvocationID: SV_DispatchThreadID,
          uint group_index: SV_GroupIndex) {

    const uint workgroup_size = uint(workgroup_size_x * workgroup_size_y);
    const int2 pixel = min(gl_GlobalInvocationID.xy, int2(merian::texture_dimensions(in_src)) - 1);

    const float own = merian::yuv_luminance(in_src[pixel].rgb);
    luminance[group_index] = own;
    if (group_index < 4)
        percentile[group_index] = 0.0; // a NaN input leaves its rank unclaimed
    GroupMemoryBarrierWithGroupSync();

    // Only four order statistics are needed, so each thread ranks its own value rather than the
    // group sorting itself.
    uint rank = 0;
    for (uint i = 0; i < workgroup_size; i++) {
        const float other = luminance[i];
        rank += uint(other < own || (other == own && i < group_index));
    }

    const float4 requested = float4(pc.firefly_percentile_lower, pc.firefly_percentile_upper,
                                    pc.adaptive_alpha_percentile_lower,
                                    pc.adaptive_alpha_percentile_upper);
    const uint4 target = uint4(clamp(round(requested * workgroup_size), 0, workgroup_size - 1));

    if (rank == target.x)
        percentile[0] = own;
    if (rank == target.y)
        percentile[1] = own;
    if (rank == target.z)
        percentile[2] = own;
    if (rank == target.w)
        percentile[3] = own;
    GroupMemoryBarrierWithGroupSync();

    if (group_index == 0) {
        quartiles.Store(gl_WorkGroupID.xy,
                        float4(percentile[0], percentile[1], percentile[2], percentile[3]));
    }
}
