#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.scene.scene;
import merian_shaders.scene.light_cdf;
import merian_shaders.utils.envmap;
import merian_shaders.scene.env_importance;
import merian_shaders.utils.morton;
import merian_shaders.utils.pseudorandom;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.colors.colorspaces;

using merian;

struct EnvImportanceBinding {
    RWStructuredBuffer<float2> cdf;
    // zeroed, see LightCdfBinding
    RWStructuredBuffer<uint> state;
    uint size;
};

struct EnvPoolBinding {
    EnvImportance importance;
    RWStructuredBuffer<uint> pool;
    uint pool_size;
    uint frame;
};

// The mean environment luminance over a stratified 2x2 within every texel, scanned in Morton order.
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
        float sum = 0.f;
        for (uint s = 0; s < 4; s++) {
            const float2 offset =
                float2(0.25f + 0.5f * float(s & 1), 0.25f + 0.5f * float(s >> 1));
            const float3 dir = equal_area_square_to_sphere((float2(texel) + offset) * inv_size);
            // the environment is looked up along -wi, and emitters may only light later bounces
            const ShadingData sd = make_env_shading_data(dir, float3(0), scene.time);
            sum += yuv_luminance(
                scene.env_map.eval(sd, scene.material_system.texture_manager, DefaultLODSampler()));
        }
        value = 0.25f * sum;
        value = isfinite(value) ? max(value, 0.f) : 0.f;
    }

    float2 total;
    const float2 prefix = group_scan(float2(value, 0.f), thread_id, total);
    const float2 before = block_prefix(params.state, block, total, thread_id);
    if (i < texels)
        params.cdf[i] = ff_add(before, prefix);
}

// Draws the pool the samplers load from, once per frame.
[shader("compute")][numthreads(64, 1, 1)] void env_pool(uint3 thread_id : SV_DispatchThreadID,
                                                        ParameterBlock<EnvPoolBinding> params) {
    const uint i = thread_id.x;
    if (i >= params.pool_size)
        return;

    if (!params.importance.is_valid()) {
        params.pool[i] = 0xFFFFFFFFu;
        return;
    }
    RandomGenerator rng = PCG(xxhash32(uint2(i, params.frame)));
    const uint2 texel = morton_decode2d(search_range_wide(
        params.importance.cdf, 0, params.importance.texel_count(), rng.next_float2()));
    params.pool[i] = texel.x | (texel.y << 16u);
}
