#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.scene.light_data;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.utils.sampling;

using merian;

// Per-frame preprocess for next event estimation: a pool of pre-resampled lights and a
// world-space grid of short per-cell lists. Everything a sample needs afterwards is one or two
// loads.

struct LightPreprocessBinding {
    StructuredBuffer<EmissiveTriangle, ScalarDataLayout> triangles;
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    // inclusive prefix sum of the flux
    StructuredBuffer<float> cdf;
    RWStructuredBuffer<uint> pool;
    // the same buffer twice: only the pool pass writes it, and a read-only binding keeps the
    // resampling loop on the read-only cache path
    RWStructuredBuffer<PooledLight, ScalarDataLayout> pool_lights_out;
    StructuredBuffer<PooledLight, ScalarDataLayout> pool_lights;
    RWStructuredBuffer<uint> grid;
    RWStructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info;
    RaytracingAccelerationStructure as;

    // fraction of the light bounds the grid spans around the camera
    float grid_coverage;
    float3 camera_position;
    uint triangle_count;
    uint pool_size;
    uint grid_candidates;
    uint grid_dimension;
    uint grid_cascades;
    float grid_cell_size;
    float grid_jitter;
    uint grid_visibility;
    uint selection;
    uint frame;
};

// Squared distance to a triangle, clamped by its own size and by the radius the query stands for
// (Conty and Kulla 2018 clamp on the node radius). Without the second clamp a light inside the
// cell takes every slot for shading points that are nowhere near it.
float clamped_distance2(const float3 to_light, const float area2, const float radius) {
    return max(dot(to_light, to_light), max(0.5f * area2, radius * radius));
}

// Importance of a light for a point over the density the pool draws it with: the pool is
// already proportional to the flux, so what remains is the geometry term towards the point,
// with the emitting side facing it. Both factors stand for the whole cell, so the cosine is
// clamped on the cell radius exactly as the distance is: a cell that straddles the light's plane
// is partly lit, and dropping it would lose those lights entirely.
float resampling_weight(const PooledLight light, const float3 pos, const float radius) {
    if (light.flux <= 0.f)
        return 0.f;
    const float3 to_pos = pos - light.centroid;
    const float area2 = length(light.scaled_normal);
    if (area2 <= 0.f)
        return 0.f;
    float height = dot(light.scaled_normal / area2, to_pos);
    if (light.is_two_sided())
        height = abs(height);
    else if (height < -radius)
        return 0.f;
    const float cos_l =
        min(max(height, 0.5f * radius) * rsqrt(max(dot(to_pos, to_pos), 1e-12f)), 1.f);
    return cos_l / clamped_distance2(-to_pos, area2, radius);
}

// Threads of the setup pass; one entry per wave would mean a wave size of one.
static const uint LIGHT_GRID_SETUP_GROUP = 256;
groupshared float3 s_setup_min[LIGHT_GRID_SETUP_GROUP];
groupshared float3 s_setup_max[LIGHT_GRID_SETUP_GROUP];
groupshared float s_setup_distance[LIGHT_GRID_SETUP_GROUP];

// Places the cascades around the camera. The outermost one has to reach the lights the camera
// can see, so its half extent comes from the distance from the camera to the farthest light,
// with the spread of the lights as a floor for a camera standing among them. Every finer cascade
// halves the cell size, so a cell covers a roughly constant angle from the camera at any scale.
[shader("compute")][numthreads(LIGHT_GRID_SETUP_GROUP, 1, 1)] void
grid_setup(uint local_id: SV_GroupIndex, ParameterBlock<LightPreprocessBinding> params) {
    const uint dim = params.grid_dimension;
    const uint cascades = max(params.grid_cascades, 1u);
    float cell = params.grid_cell_size;
    if (cell <= 0.f) {
        float3 bmin = float3(1e30f);
        float3 bmax = float3(-1e30f);
        float max_distance = 0.f;
        const uint grid_pool = min(params.pool_size, LIGHT_GRID_POOL);
        for (uint i = local_id; i < grid_pool; i += LIGHT_GRID_SETUP_GROUP) {
            const PooledLight light = params.pool_lights[i % grid_pool];
            if (!light.is_valid())
                continue;
            const float3 centroid = light.centroid;
            bmin = min(bmin, centroid);
            bmax = max(bmax, centroid);
            const float3 d = abs(centroid - params.camera_position);
            max_distance = max(max_distance, max(d.x, max(d.y, d.z)));
        }

        const uint wave = local_id / WaveGetLaneCount();
        const float3 wave_min = WaveActiveMin(bmin);
        const float3 wave_max = WaveActiveMax(bmax);
        const float wave_distance = WaveActiveMax(max_distance);
        if (WaveIsFirstLane()) {
            s_setup_min[wave] = wave_min;
            s_setup_max[wave] = wave_max;
            s_setup_distance[wave] = wave_distance;
        }
        GroupMemoryBarrierWithGroupSync();

        const uint wave_count = LIGHT_GRID_SETUP_GROUP / WaveGetLaneCount();
        for (uint w = 0; w < wave_count; w++) {
            bmin = min(bmin, s_setup_min[w]);
            bmax = max(bmax, s_setup_max[w]);
            max_distance = max(max_distance, s_setup_distance[w]);
        }

        const float extent = length(max(bmax - bmin, float3(0)));
        const float reach = params.grid_coverage * max(max_distance, 0.5f * extent);
        // the coarsest cascade spans 2 * reach, every finer one half of the one before
        cell = reach > 0.f ? max(2.f * reach / (float(dim) * exp2(float(cascades - 1u))), 1e-4f)
                           : 1.f;
    }

    for (uint cascade = local_id; cascade < cascades; cascade += LIGHT_GRID_SETUP_GROUP) {
        const float cascade_cell = cell * exp2(float(cascade));
        // snapped to the cell size so the contents do not shimmer as the camera moves
        const float3 snapped = floor(params.camera_position / cascade_cell) * cascade_cell;
        params.grid_info[cascade].origin = snapped - 0.5f * float(dim) * cascade_cell;
        params.grid_info[cascade].inv_cell_size = 1.f / cascade_cell;
        params.grid_info[cascade].dimension = dim;
        params.grid_info[cascade].cascade_count = cascades;
    }
}

// Whether the cell point sees the light, tested against one point on the triangle. Every
// triangle blocks, whichever way it faces and whatever its alpha: a point inside a wall must see
// nothing, and the tracer's culled query would see everything from there.
bool cell_sees_light(const RaytracingAccelerationStructure as,
                     const EmissiveTriangle tri,
                     const float3 from,
                     inout RandomGenerator rng) {
    const float3 bary = sample_triangle(rng);
    const float3 target = bary.x * tri.p0 + bary.y * tri.p1 + bary.z * tri.p2;
    const float3 to_target = target - from;
    const float dist = length(to_target);
    if (!(dist > 0.f))
        return true;

    RayDesc ray;
    ray.Origin = from;
    ray.Direction = to_target / dist;
    ray.TMin = 1e-3f * dist;
    ray.TMax = 0.999f * dist;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE> rq;
    rq.TraceRayInline(as, RAY_FLAG_NONE, 0xff, ray);
    rq.Proceed();
    return rq.CommittedStatus() == COMMITTED_NOTHING;
}

// Selects a light with probability flux[i] / total.
uint select_light(const StructuredBuffer<float> cdf, const uint count, const float u) {
    const float target = u * cdf[count - 1];
    uint lo = 0;
    uint hi = count - 1;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (cdf[mid] > target)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

// The pool holds independent draws from that distribution, so a sample is one load from a buffer
// small enough to stay in cache, and its density is still flux[i] / total.
[shader("compute")][numthreads(64, 1, 1)] void pool(uint3 thread_id : SV_DispatchThreadID,
                                                    ParameterBlock<LightPreprocessBinding> params) {
    const uint i = thread_id.x;
    if (i >= params.pool_size)
        return;
    if (params.triangle_count == 0 || !(params.cdf[params.triangle_count - 1] > 0.f)) {
        params.pool[i] = LIGHT_INVALID_INDEX;
        PooledLight empty;
        empty.centroid = float3(0);
        empty.flux = 0.f;
        empty.scaled_normal = float3(0);
        empty.packed_index = LIGHT_INVALID_INDEX;
        params.pool_lights_out[i] = empty;
        return;
    }

    RandomGenerator rng = PCG(xxhash32(uint2(i, params.frame)));
    const uint index = select_light(params.cdf, params.triangle_count, rng.next_float());
    params.pool[i] = index;

    if (i >= min(params.pool_size, LIGHT_GRID_POOL))
        return;

    const EmissiveLightProxy proxy = params.proxies[index];
    PooledLight light;
    light.centroid = proxy.centroid;
    light.flux = proxy.flux;
    light.scaled_normal = proxy.scaled_normal;
    light.packed_index =
        index | ((proxy.flags & uint(EmissiveTriangleFlags::EmissiveTwoSided)) != 0
                     ? PooledLight::TWO_SIDED_BIT
                     : 0u);
    params.pool_lights_out[i] = light;
}

// Threads that build one cell. Fewer than the candidate ceiling, because the group is launched
// for every cell whatever the candidate count and idle threads are not free.
static const uint LIGHT_GRID_BUILD_GROUP = 64;
static const uint LIGHT_GRID_BUILD_CHUNK = LIGHT_GRID_MAX_CANDIDATES / LIGHT_GRID_BUILD_GROUP;

// Weights of one cell's candidates as an inclusive prefix sum, and the per-wave totals the
// scan combines. One entry per wave would mean a wave size of one.
groupshared float s_prefix[LIGHT_GRID_MAX_CANDIDATES];
groupshared uint s_index[LIGHT_GRID_MAX_CANDIDATES];
groupshared float s_wave_total[LIGHT_GRID_BUILD_GROUP];

/* Each cell resamples the pool towards its own centre. A cell sample is then one load, and its
 * density is the number of slots holding that light over the slot count - no normalization to
 * carry, and exact for multiple importance sampling. The slots are drawn stratified, so they cover
 * the cell distribution instead of repeating its mode.
 *
 * One group per cell, one thread per candidate: the stratification needs the total weight before
 * it can place a slot, which a prefix sum over the group provides in log steps. The slots then
 * binary search it, and trace their visibility ray, in parallel.
 */
[shader("compute")][numthreads(LIGHT_GRID_BUILD_GROUP, 1, 1)] void
grid(uint3 group_id: SV_GroupID,
     uint local_id: SV_GroupIndex,
     ParameterBlock<LightPreprocessBinding> params) {
    const uint dim = params.grid_info[0].dimension;
    const uint cells_per_cascade = dim * dim * dim;
    const uint cell = group_id.x;
    if (dim == 0 || cell >= cells_per_cascade * params.grid_info[0].cascade_count)
        return;

    const uint cascade = cell / cells_per_cascade;
    const uint local = cell % cells_per_cascade;
    const uint3 coord = uint3(local % dim, (local / dim) % dim, local / (dim * dim));
    const LightGridInfo info = params.grid_info[cascade];
    const float cell_size = 1.f / info.inv_cell_size;
    const float3 center = info.origin + (float3(coord) + 0.5f) * cell_size;
    // the sphere the cell stands for: its circumradius, grown by the jitter the lookup adds
    const float radius = 0.5f * sqrt(3.f) * cell_size * (params.grid_jitter + 1.f);

    const uint seed = xxhash32(uint3(cell, params.frame, 0x9e3779b9u));
    // jittered per frame, so a cell is not judged from one arbitrary point forever
    RandomGenerator point_rng = PCG(seed ^ 0xc2b2ae35u);
    const float3 point = center + (point_rng.next_float3() - 0.5f) * cell_size;
    const uint candidates =
        clamp(params.grid_candidates, 1u, LIGHT_GRID_MAX_CANDIDATES);
    const uint grid_pool = max(min(params.pool_size, LIGHT_GRID_POOL), 1u);

    // 1. a contiguous chunk of candidates per thread
    const uint chunk = (candidates + LIGHT_GRID_BUILD_GROUP - 1u) / LIGHT_GRID_BUILD_GROUP;
    const uint first = local_id * chunk;

    float chunk_sum = 0.f;
    for (uint k = 0; k < chunk; k++) {
        const uint c = first + k;
        if (c >= candidates)
            break;
        RandomGenerator rng = PCG(xxhash32(uint2(seed, c)));
        const PooledLight light =
            params.pool_lights[min(uint(rng.next_float() * grid_pool), grid_pool - 1)];
        if (light.is_valid())
            chunk_sum += resampling_weight(light, point, radius);
        s_index[c] = light.is_valid() ? light.get_index() : LIGHT_INVALID_INDEX;
        // inclusive within the chunk; the chunk's own offset is added below
        s_prefix[c] = chunk_sum;
    }

    // 2. inclusive prefix sum over the candidates: scan the chunk sums, then shift each chunk
    const uint lane_count = WaveGetLaneCount();
    const uint wave = local_id / lane_count;
    const float wave_prefix = WavePrefixSum(chunk_sum);
    const float wave_sum = WaveActiveSum(chunk_sum);
    if (WaveIsFirstLane())
        s_wave_total[wave] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    const uint wave_count = (LIGHT_GRID_BUILD_GROUP + lane_count - 1u) / lane_count;
    float offset = 0.f;
    float total = 0.f;
    for (uint w = 0; w < wave_count; w++) {
        if (w == wave)
            offset = total + wave_prefix;
        total += s_wave_total[w];
    }
    for (uint k = 0; k < chunk; k++) {
        const uint c = first + k;
        if (c >= candidates)
            break;
        s_prefix[c] += offset;
    }
    GroupMemoryBarrierWithGroupSync();
    if (!(total > 0.f)) {
        if (local_id < LIGHT_GRID_SLOTS)
            params.grid[cell * LIGHT_GRID_SLOTS + local_id] = LIGHT_INVALID_INDEX;
        return;
    }
    if (local_id >= LIGHT_GRID_SLOTS)
        return;

    // 3. one slot per thread: the candidate the stratum lands in, i.e. the first prefix past it
    RandomGenerator stratum_rng = PCG(seed ^ 0x85ebca6bu);
    const float step = total / float(LIGHT_GRID_SLOTS);
    const float target = (stratum_rng.next_float() + float(local_id)) * step;
    uint lo = 0;
    uint hi = candidates - 1;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (s_prefix[mid] > target)
            hi = mid;
        else
            lo = mid + 1;
    }
    uint slot_index = s_prefix[lo] > target ? s_index[lo] : LIGHT_INVALID_INDEX;

    // 4. A near light behind a wall would otherwise take most of the cell: the distance target has
    // no way to see the wall. Dropped slots fall back to the global technique, which the density
    // charges for.
    if (params.grid_visibility != 0 && slot_index != LIGHT_INVALID_INDEX) {
        RandomGenerator rng = PCG(seed ^ (0x27d4eb2fu * (local_id + 1u)));
        if (!cell_sees_light(params.as, params.triangles[slot_index], point, rng))
            slot_index = LIGHT_INVALID_INDEX;
    }

    params.grid[cell * LIGHT_GRID_SLOTS + local_id] = slot_index;
}
