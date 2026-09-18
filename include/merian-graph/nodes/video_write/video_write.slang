import merian_shaders.colors.colorspaces;

using merian;

[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

Sampler2D<float4> in_src;

RWTexture2D<float> luma;
RWTexture2D<float2> chroma;

struct PushConstant {
    float4 matrix_0;
    float4 matrix_1;
    float4 matrix_2;
    float4 scale;
    float4 bias;
};

[vk::push_constant]
ConstantBuffer<PushConstant> params;

// One thread per chroma texel, so the 2x2 luma block it covers supplies the averaged chroma.
[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    uint2 chroma_size;
    chroma.GetDimensions(chroma_size.x, chroma_size.y);
    if (any(ipos >= int2(chroma_size)))
        return;

    uint2 luma_size;
    luma.GetDimensions(luma_size.x, luma_size.y);

    float2 cbcr = float2(0.0);
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            const int2 pos = min(ipos * 2 + int2(x, y), int2(luma_size) - 1);

            float3 rgb = in_src[pos].rgb;
            if (params.bias.w != 0.0)
                rgb = rgb_to_srgb(saturate(rgb));

            float3 ycbcr = params.matrix_0.xyz * rgb.r + params.matrix_1.xyz * rgb.g +
                           params.matrix_2.xyz * rgb.b;
            ycbcr = ycbcr * params.scale.xyz + params.bias.xyz;

            luma[pos] = ycbcr.x;
            cbcr += ycbcr.yz;
        }
    }

    chroma[ipos] = cbcr * 0.25;
}
