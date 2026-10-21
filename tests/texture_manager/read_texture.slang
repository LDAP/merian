import merian_shaders.utils.texture_manager;

struct Params {
    ParameterBlock<merian::TextureManager> tm;
    RWStructuredBuffer<uint> output;
    uint texture_id;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(ParameterBlock<Params> params) {
    // Sample at center (0.5, 0.5) — for a 1x1 texture this gives the exact texel
    const float4 color =
        params.tm.get_textures().get_sampler2D(params.texture_id).SampleLevel(float2(0.5, 0.5), 0);
    const float4 prev_color = params.tm.get_textures(true)
                                  .get_sampler2D(params.texture_id)
                                  .SampleLevel(float2(0.5, 0.5), 0);
    for (uint i = 0; i < 4; i++) {
        params.output[i] = asuint(color[i]);
        params.output[4 + i] = asuint(prev_color[i]);
    }
}
