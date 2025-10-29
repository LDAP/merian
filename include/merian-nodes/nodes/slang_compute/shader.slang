#include "config.h"

import merian_shaders.utils.textures;
import merian_shaders.colors.tonemaps;

using merian;

[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;
[vk::constant_id(2)]
const int tonemap = 0;
[vk::constant_id(3)]
const int alpha_mode = 0;
[vk::constant_id(4)]
const int clamp_output = 1;

struct PushConstant {
    float param1;
    float param2;
    float param3;
    float param4;
    float param5;

    float perceptual_exp;
};

[vk::push_constant]
ConstantBuffer<PushConstant> params;

struct MerianIn {
    [vk::binding(0)]
    Sampler2D<float4> img_src;
};

struct MerianOut {
    [vk::binding(1)]
    RWTexture2D<float4> img_output;
};

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID, MerianIn merian_in, MerianOut merian_out) {
    if (not_on_texture(ipos,  merian_out.img_output))
        return;

    const float4 rgb_in =  merian_in.img_src.Load(int3(ipos, 0));
    float3 rgb_out;

    switch (tonemap) {
    case TONEMAP_NONE:
        rgb_out = rgb_in.rgb;
        break;
    case TONEMAP_CLAMP:
        rgb_out = tonemap_clamp(rgb_in.rgb);
        break;
    case TONEMAP_UNCHARTED_2:
        rgb_out = tonemap_uncharted2(rgb_in.rgb, params.param1, params.param2);
        break;
    case TONEMAP_REINHARD_EXTENDED:
        rgb_out = tonemap_reinhard(rgb_in.rgb, params.param1);
        break;
    case TONEMAP_ACES:
        rgb_out = tonemap_aces(rgb_in.rgb);
        break;
    case TONEMAP_ACES_APPROX:
        rgb_out = tonemap_aces_approx(rgb_in.rgb, params.param1, params.param2, params.param3,
                                      params.param4, params.param5);
        break;
    case TONEMAP_LOTTES:
        rgb_out = tonemap_lottes(rgb_in.rgb, params.param1, params.param2, params.param3,
                                 params.param4, params.param5);
        break;
    }

    if (clamp_output != 0)
        rgb_out = clamp(rgb_out, 0, 1);

    switch (alpha_mode) {
    case ALPHA_MODE_PASSTHROUGH:
        merian_out.img_output.Store(ipos, float4(rgb_out, rgb_in.a));
        break;
    case ALPHA_MODE_LUMINANCE:
         merian_out.img_output.Store(ipos, float4(rgb_out, yuv_luminance(rgb_out)));
        break;
    case ALPHA_MODE_PERCEPTUAL_LUMINANCE:
         merian_out.img_output.Store(ipos,
                         float4(rgb_out, pow(yuv_luminance(rgb_out), 1. / params.perceptual_exp)));
        break;
    }
}
