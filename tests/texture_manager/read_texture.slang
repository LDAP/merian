import merian_shaders.utils.texture_manager;

struct Params {
    ParameterBlock<merian::TextureManager> tm;
    RWStructuredBuffer<uint> output;
    uint texture_id;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    Sampler2D<float4> tex = params.tm.textures[params.texture_id];
    // Sample at center (0.5, 0.5) — for a 1x1 texture this gives the exact texel
    float4 color = tex.SampleLevel(float2(0.5, 0.5), 0);
    params.output[0] = asuint(color.r);
    params.output[1] = asuint(color.g);
    params.output[2] = asuint(color.b);
    params.output[3] = asuint(color.a);
}
