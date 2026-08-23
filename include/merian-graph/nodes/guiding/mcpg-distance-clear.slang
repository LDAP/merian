import merian_shaders.utils.textures;
import mc_distance;

using merian;

struct DistanceClearBinding {
    DistanceMCGrid grid;
};

// The grid is transient: the projection rebuilds it every frame.
[shader("compute")][numthreads(8, 8, 1)] void main(uint3 thread_id : SV_DispatchThreadID,
                                                   ParameterBlock<DistanceClearBinding> params) {
    for (uint level = 0; level < distance_mc_level_count; level++) {
        if (all(thread_id.xy < texture_dimensions(params.grid.levels[level])))
            params.grid.levels[level][thread_id.xy] = float4(0);
    }
}
