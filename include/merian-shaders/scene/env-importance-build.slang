import merian_shaders.scene.scene;
import merian_shaders.scene.light_cdf;
import merian_shaders.utils.envmap;
import merian_shaders.utils.morton;
import merian_shaders.colors.colorspaces;

using merian;

struct EnvImportanceBinding {
    RWStructuredBuffer<float2> cdf;
    // zeroed, see LightCdfBinding
    RWStructuredBuffer<uint> state;
    uint size;
    uint samples;
};

// The mean environment luminance over a stratified grid in every texel, scanned in Morton order.
[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
build(uint thread_id : SV_GroupIndex,
      ParameterBlock<Scene> scene,
      ParameterBlock<EnvImportanceBinding> params) {
    const uint block = start_block(params.state, thread_id);
    const uint i = block * CDF_GROUP_SIZE + thread_id;
    const uint texels = params.size * params.size;

    float value = 0.f;
    if (i < texels) {
        const uint2 texel = morton_decode2d(i);
        const float inv_size = 1.f / float(params.size);
        const float inv_samples = 1.f / float(params.samples);
        float sum = 0.f;
        for (uint y = 0; y < params.samples; y++) {
            for (uint x = 0; x < params.samples; x++) {
                const float2 offset = (float2(x, y) + 0.5f) * inv_samples;
                const float3 dir = equal_area_square_to_sphere((float2(texel) + offset) * inv_size);
                // the environment is looked up along -wi, and emitters may only light later bounces
                const ShadingData sd = make_env_shading_data(dir, float3(0), scene.time);
                const float luminance = yuv_luminance(scene.eval_env_map(sd, DefaultLODSampler()));
                sum += isfinite(luminance) ? max(luminance, 0.f) : 0.f;
            }
        }
        value = sum * inv_samples * inv_samples;
    }

    float2 total;
    const float2 prefix = group_scan(float2(value, 0.f), thread_id, total);
    const float2 before = block_prefix(params.state, block, total, thread_id);
    if (i < texels)
        params.cdf[i] = ff_add(before, prefix);
}
