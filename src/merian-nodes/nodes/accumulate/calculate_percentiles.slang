import merian_shaders.utils.textures;
import merian_shaders.colors.colorspaces;

[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

struct PushConstant {
    float firefly_percentile_lower : packoffset(c0);
    float firefly_percentile_upper : packoffset(c0.y);
    float adaptive_alpha_percentile_lower : packoffset(c0.z);
    float adaptive_alpha_percentile_upper : packoffset(c0.w);
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

[vk::binding(0)]
Sampler2D<float4> img_src;

[vk::binding(0, 1)]
RWTexture2D<float4> img_quartiles;

groupshared float sort[workgroup_size_x * workgroup_size_y];

uint2 spvTextureSize(Texture2D<float4> Tex, uint Level, out uint Param) {
    uint2 ret;
    Tex.GetDimensions(Level, ret.x, ret.y, Param);
    return ret;
}

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
void main(uint3 gl_WorkGroupID: SV_GroupID,
          uint3 gl_GlobalInvocationID: SV_DispatchThreadID,
          uint group_index: SV_GroupIndex) {

    const int workgroup_size = workgroup_size_x * workgroup_size_y;
    int2 pixel = min(gl_GlobalInvocationID.xy, merian::texture_size(img_src, 0));

    // Sort luminance using odd even sort
    sort[group_index] = merian::yuv_luminance(img_src.Load(int3(pixel, 0)).rgb);
    GroupMemoryBarrierWithGroupSync();

    if (group_index < workgroup_size - 1) {
        for (int i = 0; i < workgroup_size / 2; i++) {
            if (group_index % 2 == 0) {
                if (sort[group_index] > sort[group_index + 1]) {
                    const float tmp = sort[group_index];
                    sort[group_index] = sort[group_index + 1];
                    sort[group_index + 1] = tmp;
                }
            }
            GroupMemoryBarrierWithGroupSync();

            if (group_index % 2 == 1) {
                if (sort[group_index] > sort[group_index + 1]) {
                    const float tmp = sort[group_index];
                    sort[group_index] = sort[group_index + 1];
                    sort[group_index + 1] = tmp;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (group_index == 0) {
        const float firefly_lower = sort[uint(
            clamp(round(pc.firefly_percentile_lower * workgroup_size), 0, workgroup_size - 1))];
        const float firefly_upper = sort[uint(
            clamp(round(pc.firefly_percentile_upper * workgroup_size), 0, workgroup_size - 1))];
        const float adaptive_alpha_lower = sort[uint(clamp(
            round(pc.adaptive_alpha_percentile_lower * workgroup_size), 0, workgroup_size - 1))];
        const float adaptive_alpha_upper = sort[uint(clamp(
            round(pc.adaptive_alpha_percentile_upper * workgroup_size), 0, workgroup_size - 1))];

        img_quartiles.Store(gl_WorkGroupID.xy, float4(firefly_lower, firefly_upper,
                                                      adaptive_alpha_lower, adaptive_alpha_upper));
    }
}
