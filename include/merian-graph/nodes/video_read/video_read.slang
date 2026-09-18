import merian_shaders.colors.colorspaces;
import merian_shaders.utils.textures;

using merian;

[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

Sampler2D<float> luma;
Sampler2D<float2> chroma;

RWTexture2D<float4> out_out;

struct PushConstant {
    float4 matrix_0;
    float4 matrix_1;
    float4 matrix_2;
    float4 scale;
    float4 bias;
};

[vk::push_constant]
ConstantBuffer<PushConstant> params;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, out_out))
        return;

    uint2 size;
    out_out.GetDimensions(size.x, size.y);
    const float2 uv = (float2(ipos) + 0.5) / float2(size);

    // chroma is subsampled, so both planes are sampled in normalized coordinates
    float3 ycbcr = float3(luma.SampleLevel(uv, 0), chroma.SampleLevel(uv, 0));
    ycbcr = ycbcr * params.scale.xyz + params.bias.xyz;

    float3 rgb = params.matrix_0.xyz * ycbcr.x + params.matrix_1.xyz * ycbcr.y +
                 params.matrix_2.xyz * ycbcr.z;
    if (params.bias.w != 0.0)
        rgb = srgb_to_rgb(saturate(rgb));

    out_out[ipos] = float4(rgb, 1.0);
}
