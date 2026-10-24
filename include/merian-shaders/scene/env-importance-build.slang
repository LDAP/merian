import merian_shaders.colors.colorspaces;
import merian_shaders.scene.env_importance;
import merian_shaders.scene.light_cdf;
import merian_shaders.scene.light_data;
import merian_shaders.scene.scene;
import merian_shaders.utils.envmap;
import merian_shaders.utils.morton;

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

struct EnvLeavesBinding {
    StructuredBuffer<float2> texel_cdf;
    RWStructuredBuffer<uint> keys;
    RWStructuredBuffer<float2> cdf;
    RWStructuredBuffer<uint> lookup;
    // zeroed, see LightCdfBinding
    RWStructuredBuffer<uint> state;
    uint size;
    uint budget;
};

// the largest blocks inside a base texel that hold at most 1 / budget of the luminance
[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
leaves(uint thread_id : SV_GroupIndex, ParameterBlock<EnvLeavesBinding> params) {
    const uint block = start_block(params.state, thread_id);
    const uint texel = block * CDF_GROUP_SIZE + thread_id;
    const uint texels = params.size * params.size;
    const uint base_bits = env_grid_shift(params.size, ENV_IMPORTANCE_BASE_LOG2);
    const uint lookup_bits = env_grid_shift(params.size, ENV_LOOKUP_LOG2);

    uint bits = 0;
    if (texel < texels) {
        const float2 sum = params.texel_cdf[texels - 1];
        const float limit = (sum.x + sum.y) / float(params.budget);
        for (bits = base_bits; bits > 0; bits -= 2u) {
            const uint first = (texel >> bits) << bits;
            const float2 before = first > 0 ? params.texel_cdf[first - 1] : float2(0);
            if (!(ff_diff(params.texel_cdf[first + (1u << bits) - 1u], before) > limit))
                break;
        }
    }
    const bool starts = texel < texels && (texel & ((1u << bits) - 1u)) == 0;

    float2 total;
    const float2 prefix = group_scan(float2(starts ? 1.f : 0.f, 0.f), thread_id, total);
    const float2 count = ff_add(block_prefix(params.state, block, total, thread_id), prefix);
    const uint leaf = uint(count.x + count.y) - 1u;
    if (starts) {
        params.keys[leaf] = texel;
        params.cdf[leaf] = params.texel_cdf[texel + (1u << bits) - 1u];
    }
    if (texel < texels && (texel & ((1u << lookup_bits) - 1u)) == 0)
        params.lookup[texel >> lookup_bits] = leaf;
    if (texel == texels - 1u) {
        params.keys[leaf + 1u] = texels;
        params.lookup[texels >> lookup_bits] = leaf + 1u;
    }
}

struct EnvGuideBinding {
    StructuredBuffer<float2> cdf;
    StructuredBuffer<uint> lookup;
    RWStructuredBuffer<uint> guide;
    uint lookup_cells;
    uint buckets;
};

[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
leaf_guide(uint bucket : SV_DispatchThreadID, ParameterBlock<EnvGuideBinding> params) {
    write_guide(params.cdf, params.guide, params.lookup[params.lookup_cells], params.buckets,
                bucket);
}
