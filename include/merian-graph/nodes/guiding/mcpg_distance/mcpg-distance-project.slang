import merian_shaders.scene.scene;
import merian_shaders.utils.random;
import merian_shaders.utils.hash;
import merian_shaders.utils.textures;
import "merian-graph/nodes/guiding/mcpg_distance/mc_distance.slang";

using merian;

struct DistanceProjectBinding {
    DistanceMCGrid grid;
    Sampler2D<float4> prev;
    uint level;
    uint2 dim;
};

// Concurrent writes to the same target vertex are benign: either state is a valid estimate.
[shader("compute")][numthreads(8, 8, 1)] void main(uint3 thread_id : SV_DispatchThreadID,
                                                   ParameterBlock<Scene> scene,
                                                   ParameterBlock<DistanceProjectBinding> params) {
    const uint level = params.level;
    const uint2 idx = thread_id.xy;
    const uint2 cells = texture_dimensions(params.grid.levels[level]);
    if (any(idx >= cells))
        return;

    const DistanceMCState s = DistanceMCState::unpack(params.prev.Load(int3(idx, level)));
    if (!(s.sum_w > 0.0))
        return;

    const float2 pixel = params.grid.vertex_pixel(idx, level);
    const float mean = s.moments.x / s.sum_w;
    const float3 pos = scene.prev_camera.position +
                       scene.prev_camera.get_pinhole_direction_unjittered(pixel, params.dim) * mean;
    const float2 new_pixel = scene.camera.project_pinhole(pos, params.dim);

    const float width = distance_mc_base_width * float(1u << level);
    const int2 target = int2(round(new_pixel / width + 0.5));
    if (any(target < 0) || any(target >= int2(cells)))
        return;

    DistanceMCState moved = s;
    distance_mc_state_shift(moved, distance(pos, scene.camera.position) - mean);
    params.grid.levels[level][uint2(target)] = moved.pack();
}
