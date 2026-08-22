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
    // inclusive prefix sum of the flux
    StructuredBuffer<float> cdf;
    RWStructuredBuffer<uint> pool;
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
    float grid_cell_size;
    uint grid_visibility;
    uint selection;
    uint frame;
};

// Squared distance to a triangle, clamped by its own size and by the radius the query stands for
// (Conty and Kulla 2018 clamp on the node radius). Without the second clamp a light inside the
// cell takes every slot for shading points that are nowhere near it.
float clamped_distance2(const EmissiveTriangle tri, const float3 pos, const float radius) {
    const float3 centroid = (tri.p0 + tri.p1 + tri.p2) * (1.f / 3.f);
    const float3 to_light = centroid - pos;
    const float3 scaled_n = tri.get_scaled_normal();
    return max(dot(to_light, to_light), max(0.5f * length(scaled_n), radius * radius));
}

// Importance of a light for a point over the density the pool draws it with: the pool is
// already proportional to the flux, so what remains is the geometry term towards the point,
// with the emitting side facing it.
float resampling_weight(const EmissiveTriangle tri, const float3 pos, const float radius) {
    if (tri.flux <= 0.f)
        return 0.f;
    const float3 centroid = (tri.p0 + tri.p1 + tri.p2) * (1.f / 3.f);
    const float3 scaled_n = tri.get_scaled_normal();
    const float3 to_pos = pos - centroid;
    const float area2 = length(scaled_n);
    if (area2 <= 0.f)
        return 0.f;
    float cos_l = dot(scaled_n / area2, to_pos) * rsqrt(max(dot(to_pos, to_pos), 1e-12f));
    if (tri.is_two_sided())
        cos_l = abs(cos_l);
    if (cos_l <= 0.f)
        return 0.f;
    return cos_l / clamped_distance2(tri, pos, radius);
}

// Places the grid around the camera, sized so that it spans a fraction of the lit region. The
// extent comes from a stride over the pool: the scene itself may not know its own bounds.
[shader("compute")][numthreads(1, 1, 1)] void
grid_setup(ParameterBlock<LightPreprocessBinding> params) {
    const uint dim = params.grid_dimension;
    float cell = params.grid_cell_size;
    if (cell <= 0.f) {
        float3 bmin = float3(1e30f);
        float3 bmax = float3(-1e30f);
        const uint samples = min(params.pool_size, 512u);
        const uint stride = max(params.pool_size / max(samples, 1u), 1u);
        for (uint i = 0; i < samples; i++) {
            const uint index = params.pool[i * stride];
            if (index == LIGHT_INVALID_INDEX)
                continue;
            const EmissiveTriangle tri = params.triangles[index];
            const float3 centroid = (tri.p0 + tri.p1 + tri.p2) * (1.f / 3.f);
            bmin = min(bmin, centroid);
            bmax = max(bmax, centroid);
        }
        const float extent = length(max(bmax - bmin, float3(0)));
        cell = extent > 0.f ? max(params.grid_coverage * extent / float(dim), 1e-4f) : 1.f;
    }

    // snapped to the cell size so the contents do not shimmer as the camera moves
    const float3 snapped = floor(params.camera_position / cell) * cell;
    params.grid_info[0].origin = snapped - 0.5f * float(dim) * cell;
    params.grid_info[0].inv_cell_size = 1.f / cell;
    params.grid_info[0].dimension = dim;
}

// Whether the cell point sees the light, tested against one point on the triangle.
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

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;
    rq.TraceRayInline(as, RAY_FLAG_NONE, 0xff, ray);
    rq.Proceed();
    return rq.CommittedStatus() == COMMITTED_NOTHING;
}

// Selects a light with probability weights[i] / total.
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
        return;
    }

    RandomGenerator rng = PCG(xxhash32(uint2(i, params.frame)));
    params.pool[i] = select_light(params.cdf, params.triangle_count, rng.next_float());
}

    // Each cell resamples the pool towards its own centre. A cell sample is then one load, and its
    // density is the number of slots holding that light over the slot count — no normalization to
    // carry, and exact for multiple importance sampling. The slots are drawn stratified, so they
    // cover the cell distribution instead of repeating its mode; the same candidates are walked
    // twice because their total weight is only known at the end of the first walk.
    [shader("compute")][numthreads(64, 1, 1)] void grid(
        uint3 thread_id : SV_DispatchThreadID, ParameterBlock<LightPreprocessBinding> params) {
    const uint dim = params.grid_info[0].dimension;
    const uint cell = thread_id.x;
    if (dim == 0 || cell >= dim * dim * dim)
        return;

    const uint3 coord = uint3(cell % dim, (cell / dim) % dim, cell / (dim * dim));
    const float cell_size = 1.f / params.grid_info[0].inv_cell_size;
    const float3 center = params.grid_info[0].origin + (float3(coord) + 0.5f) * cell_size;
    const float radius = 0.5f * cell_size;

    const uint seed = xxhash32(uint3(cell, params.frame, 0x9e3779b9u));
    // jittered per frame, so a cell is not judged from one arbitrary point forever
    RandomGenerator point_rng = PCG(seed ^ 0xc2b2ae35u);
    const float3 point = center + (point_rng.next_float3() - 0.5f) * cell_size;
    const uint candidates = max(params.grid_candidates, 1u);
    const uint pool_size = max(params.pool_size, 1u);

    float total = 0.f;
    {
        RandomGenerator rng = PCG(seed);
        for (uint c = 0; c < candidates; c++) {
            const uint index = params.pool[min(uint(rng.next_float() * pool_size), pool_size - 1)];
            if (index == LIGHT_INVALID_INDEX)
                continue;
            total += resampling_weight(params.triangles[index], point, radius);
        }
    }
    if (!(total > 0.f)) {
        for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++)
            params.grid[cell * LIGHT_GRID_SLOTS + slot] = LIGHT_INVALID_INDEX;
        return;
    }

    RandomGenerator stratum_rng = PCG(seed ^ 0x85ebca6bu);
    const float step = total / float(LIGHT_GRID_SLOTS);
    float next = stratum_rng.next_float() * step;
    uint slot = 0;
    float sum = 0.f;
    RandomGenerator rng = PCG(seed);
    for (uint c = 0; c < candidates && slot < LIGHT_GRID_SLOTS; c++) {
        const uint index = params.pool[min(uint(rng.next_float() * pool_size), pool_size - 1)];
        if (index == LIGHT_INVALID_INDEX)
            continue;
        sum += resampling_weight(params.triangles[index], point, radius);
        while (sum > next && slot < LIGHT_GRID_SLOTS) {
            params.grid[cell * LIGHT_GRID_SLOTS + slot] = index;
            slot++;
            next += step;
        }
    }
    for (; slot < LIGHT_GRID_SLOTS; slot++)
        params.grid[cell * LIGHT_GRID_SLOTS + slot] = LIGHT_INVALID_INDEX;

    // A near light behind a wall would otherwise take most of the cell: the distance target has no
    // way to see the wall. Dropped slots fall back to the global technique, which the density
    // charges for.
    if (params.grid_visibility != 0) {
        for (uint s = 0; s < LIGHT_GRID_SLOTS; s++) {
            const uint index = params.grid[cell * LIGHT_GRID_SLOTS + s];
            if (index == LIGHT_INVALID_INDEX)
                continue;
            if (!cell_sees_light(params.as, params.triangles[index], point, rng))
                params.grid[cell * LIGHT_GRID_SLOTS + s] = LIGHT_INVALID_INDEX;
        }
    }
}
