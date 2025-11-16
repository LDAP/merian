#include "config.h"

import merian_shaders.utils.textures;

using merian;

struct StaticDims { int x; int y; int z; };
struct AsInputImage { String name; };

[__AttributeUsage(_AttributeTargets.Var)]
struct MerianExtentStaticAttribute
{
    int x;
    int y;
    int z;
};

[__AttributeUsage(_AttributeTargets.Var)]
struct MerianExtentAsAttribute
{
    String name;
};

struct NodeIn {
    [vk::binding(0)]
    Sampler2D<float4> img_src;
};

struct NodeOut {
    [vk::binding(1)] [MerianExtentAs("img_src")]
    RWTexture2D<float4> img_output;
};

[numthreads(16, 16, 1)]
[shader("compute")]
void main(int2 ipos: SV_DispatchThreadID, uniform NodeIn node_in, uniform NodeOut node_out) {
    if (not_on_texture(ipos,  node_out.img_output))
        return;

    const float4 rgb_in =  node_in.img_src.Load(int3(ipos, 0));
    float3 rgb_out;

    rgb_out = float3(1.0) - clamp(rgb_in.rgb, float3(0), float3(1.0));

    node_out.img_output.Store(ipos, float4(rgb_out, rgb_in.a));
}
