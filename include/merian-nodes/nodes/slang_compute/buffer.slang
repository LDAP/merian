import merian_shaders.utils.textures;
import reflection_attributes;

using merian;

struct NodeIn {
    [vk::binding(0)]
    Sampler2D<float4> img_src;
};

struct NodeOut {
    [vk::binding(1)] [MerianExtentAs("img_src")]
    RWTexture2D<float4> img_output;
};

struct NodeProperties {
    float4 color;
    String str;
    [MerianFloatRange(0.0, 1.5)]
    float f;
    [MerianIntRange(0, 5)]
    int i;
    [MerianIntRange(1, 6)]
    uint ui;
}

[numthreads(16, 16, 1)]
[shader("compute")]
[MerianOperateOn("img_src")]
void main(int2 ipos: SV_DispatchThreadID,
        uniform NodeIn node_in,
        uniform NodeOut node_out,
        uniform NodeProperties node_props) {
    if (not_on_texture(ipos,  node_out.img_output))
        return;

    const float4 rgb_in =  node_in.img_src.Load(int3(ipos, 0));
    float3 rgb_out;

    rgb_out = float3(1.0) - clamp(rgb_in.rgb, float3(0), float3(1.0));

    node_out.img_output.Store(ipos, float4(rgb_out, rgb_in.a));

}
