#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.scene.scene;
import merian_shaders.utils.envmap;
import merian_shaders.utils.morton;
import merian_shaders.utils.pseudorandom;
import merian_shaders.colors.colorspaces;

using merian;

struct EnvImportanceBinding {
    RWStructuredBuffer<float4> levels;
    uint size;
    uint level;
    // texels per side of the level being written
    uint level_size;
};

// First quad of a level; every level holds its texels in Morton order, four per entry.
uint level_offset(const uint size, const uint level) {
    const uint quads = (size * size) / 4u;
    return (4u * (quads - (quads >> (2u * level)))) / 3u;
}

uint quad_index(const uint2 texel) {
    return morton_encode2d(texel >> 1);
}

uint quad_child(const uint2 texel) {
    return (texel.x & 1u) | ((texel.y & 1u) << 1u);
}

// Level 0: mean environment luminance over a stratified 2x2 within the texel.
[shader("compute")][numthreads(8, 8, 1)] void build(uint3 thread_id : SV_DispatchThreadID,
                                                    ParameterBlock<Scene> scene,
                                                    ParameterBlock<EnvImportanceBinding> params) {
    if (any(thread_id.xy >= params.size))
        return;

    const float inv_size = 1.f / float(params.size);
    float sum = 0.f;
    for (uint s = 0; s < 4; s++) {
        const float2 offset = float2(0.25f + 0.5f * float(s & 1), 0.25f + 0.5f * float(s >> 1));
        const float3 dir = equal_area_square_to_sphere((float2(thread_id.xy) + offset) * inv_size);
        // the environment is looked up along -wi, and emitters may only light later bounces
        const ShadingData sd = make_env_shading_data(dir, float3(0), scene.time);
        sum += yuv_luminance(
            scene.env_map.eval(sd, scene.material_system.texture_manager, DefaultLODSampler()));
    }

    const float value = 0.25f * sum;
    params.levels[thread_id.y * params.size + thread_id.x] =
        isfinite(value) ? max(value, 0.f) : 0.f;
}

    // Coarser levels: the mean of the four finer texels.
    [shader("compute")][numthreads(8, 8, 1)] void reduce(
        uint3 thread_id : SV_DispatchThreadID, ParameterBlock<EnvImportanceBinding> params) {
    if (any(thread_id.xy >= params.level_size))
        return;

    // the four children are one quad at the finer level
    const float4 children =
        params.levels[level_offset(params.size, params.level - 1) + morton_encode2d(thread_id.xy)];
    const float mean = 0.25f * (children.x + children.y + children.z + children.w);

    params.levels[level_offset(params.size, params.level) + quad_index(thread_id.xy)]
                 [quad_child(thread_id.xy)] = mean;
}
