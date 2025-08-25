[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

[vk::binding(0)]
Texture2D<float4> img_src;
[vk::binding(0)]
SamplerState img_src_sampler;
[vk::binding(1)]
RWTexture2D<float4> img_output;

import merian_shaders.utils.textures;

using merian;

#define FXAA_PC 1
#define FXAA_HLSL_5 1
#define FXAA_QUALITY__PRESET 23
#define FXAA_GATHER4_ALPHA 0

#include "fxaa.h"

struct PushConstant {
    int enable;
    float fxaaQualitySubpix;
    float fxaaQualityEdgeThreshold;
    float fxaaQualityEdgeThresholdMin;
};

[vk::push_constant]
ConstantBuffer<PushConstant> params;

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID) {
    if (not_on_texture(ipos, img_output))
        return;

    const float2 size = texture_dimensions(img_output);
    const float2 pos = (float2(ipos) + 0.5) / size;

    float4 output_color;
    if (params.enable != 0)
        output_color = FxaaPixelShader(
            pos,       // center of pixel
            float4(0), // Used only for FXAA Console; {xy__} = upper left of pixel {__zw} = lower
                       // right of pixel
            { img_src_sampler,
              img_src }, // Input color texture, {rgb_} = color in linear or perceptual color space,
                         // {___a} = luma in perceptual color space (not linear)
            { img_src_sampler, img_src }, // Only used on the optimized 360 version of FXAA Console
            { img_src_sampler, img_src }, // Only used on the optimized 360 version of FXAA Console
            1. / size,                    // Only used on FXAA Quality.
            float4(0),                    // Only used on FXAA Console.
            float4(0),                    // Only used on FXAA Console.
            float4(0),                    // Only used on FXAA Console.
            params.fxaaQualitySubpix, params.fxaaQualityEdgeThreshold,
            params.fxaaQualityEdgeThresholdMin,
            8.0,      // Only used on FXAA Console.
            0.125,    // Only used on FXAA Console.
            0.05,     // Only used on FXAA Console.
            float4(0) // Extra constants for 360 FXAA Console only.
        );
    else
        output_color = img_src.Load(int3(ipos, 0));

    img_output.Store(ipos, output_color);
}
