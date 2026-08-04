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

groupshared float sort[workgroup_size_x * workgroup_size_y];

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(uint3 gl_WorkGroupID: SV_GroupID,
          uint3 gl_GlobalInvocationID: SV_DispatchThreadID,
          uint group_index: SV_GroupIndex) {

    const uint n = uint(workgroup_size_x * workgroup_size_y);
    const int2 pixel = min(gl_GlobalInvocationID.xy, int2(merian::texture_dimensions(in_src)) - 1);
    sort[group_index] = merian::yuv_luminance(in_src[pixel].rgb);
    GroupMemoryBarrierWithGroupSync();

    // Bitonic sorting network; n must be a power of two. Only the lower thread of each compared
    // pair (ixj > group_index) writes both slots, so a stage has no intra-slot race.
    for (uint k = 2; k <= n; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint ixj = group_index ^ j;
            if (ixj > group_index) {
                const float a = sort[group_index];
                const float b = sort[ixj];
                const bool ascending = (group_index & k) == 0;
                if ((a > b) == ascending) {
                    sort[group_index] = b;
                    sort[ixj] = a;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (group_index == 0) {
        const float firefly_lower =
            sort[uint(clamp(round(pc.firefly_percentile_lower * n), 0, n - 1))];
        const float firefly_upper =
            sort[uint(clamp(round(pc.firefly_percentile_upper * n), 0, n - 1))];
        const float adaptive_alpha_lower =
            sort[uint(clamp(round(pc.adaptive_alpha_percentile_lower * n), 0, n - 1))];
        const float adaptive_alpha_upper =
            sort[uint(clamp(round(pc.adaptive_alpha_percentile_upper * n), 0, n - 1))];

        quartiles.Store(gl_WorkGroupID.xy, float4(firefly_lower, firefly_upper,
                                                  adaptive_alpha_lower, adaptive_alpha_upper));
    }
}
