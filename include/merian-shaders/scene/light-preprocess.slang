#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.scene.env_importance;
import merian_shaders.scene.light_data;
import merian_shaders.utils.hash;
import merian_shaders.utils.random;
import merian_shaders.utils.sampling;

using merian;

// Per-frame preprocess for next event estimation: the share the environment takes of the light
// samples, a pool drawn from the flux distribution, and a world-space grid of per-cell lists.

struct LightPreprocessBinding {
    StructuredBuffer<EmissiveTriangle, ScalarDataLayout> triangles;
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    // inclusive prefix sum of the flux
    StructuredBuffer<float> cdf;
    RWStructuredBuffer<uint> pool;
    StructuredBuffer<LightGeometry, ScalarDataLayout> light_geometries;
    StructuredBuffer<LightRemap, ScalarDataLayout> light_remap;
    uint light_remap_count;
    RWStructuredBuffer<LightSource, ScalarDataLayout> sources_out;
    StructuredBuffer<LightSource, ScalarDataLayout> sources;
    RWStructuredBuffer<uint> source_of_out;
    StructuredBuffer<uint> source_of;
    RWStructuredBuffer<float> source_cdf_out;
    RWStructuredBuffer<uint> pool_source_out;
    StructuredBuffer<uint> pool_source;
    uint light_geometry_count;
    RWStructuredBuffer<uint> grid;
    RWStructuredBuffer<float> grid_key;
    RWStructuredBuffer<float> grid_visibility;
    RWStructuredBuffer<LightSource, ScalarDataLayout> grid_slot_sources;
    RWStructuredBuffer<float> grid_probability;
    StructuredBuffer<float> grid_visibility_prev;
    RWStructuredBuffer<uint> grid_feedback;
    StructuredBuffer<uint> grid_feedback_prev;
    RWStructuredBuffer<uint> grid_discovered;
    StructuredBuffer<uint> grid_discovered_prev;
    StructuredBuffer<uint> grid_prev;
    StructuredBuffer<float> grid_key_prev;
    RWStructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info;
    StructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info_prev;

    // fraction of the light bounds the grid spans around the camera
    float grid_coverage;
    float3 camera_position;
    uint triangle_count;
    uint pool_size;
    uint grid_new_lights;
    uint grid_dimension;
    uint grid_cascades;
    float grid_cell_size;
    float grid_jitter;
    // in cells of the finest cascade
    float grid_source_extent;
    // frames over which a key that stopped earning its slot is overtaken
    uint grid_max_age;
    bool grid_a_res;
    float grid_even_share;
    LightSlotWeighing slot_weighing;
    bool env_regions;
    EnvImportance env_importance;
    StructuredBuffer<float> env_split;
    // drop everything carried over, for a frame the previous grid cannot describe
    bool grid_reset;
    uint frame;
};

struct EnvSplitBinding {
    // inclusive prefix sum of the triangle flux
    StructuredBuffer<float> cdf;
    EnvImportance env_importance;
    RWStructuredBuffer<float> env_split;
    uint triangle_count;
    // 0 keeps the manual share
    float scene_radius;
    float manual_probability;
    uint from_power;
    float min_probability;
    float max_probability;
};

/* The share of the light samples the environment takes, from the power each side emits.
 *
 * The environment's power is the flux an infinite light of that mean radiance puts through the
 * scene's bounding sphere, 4 pi^2 r^2 L (pbrt-v4, ImageInfiniteLight::Phi). It is blind to
 * occlusion, which the limits bound.
 */
[shader("compute")][numthreads(1, 1, 1)] void env_split(ParameterBlock<EnvSplitBinding> params) {
    const float triangles =
        params.triangle_count > 0 ? max(params.cdf[params.triangle_count - 1], 0.f) : 0.f;
    const float radiance = params.env_importance.get_average();
    const float env = 4.f * MERIAN_PI * MERIAN_PI * params.scene_radius * params.scene_radius *
                      max(radiance, 0.f);

    float p = params.manual_probability;
    if (params.from_power != 0u && params.scene_radius > 0.f && env + triangles > 0.f) {
        p = env / (env + triangles);
        if (env > 0.f && triangles > 0.f) {
            p = clamp(p, params.min_probability, params.max_probability);
        }
    }
    params.env_split[0] = clamp(p, 0.f, 1.f);
}

// Over the pool's density for the region, on the scale of the triangles' flux times geometry term.
float env_region_weight(const LightPreprocessBinding params) {
    const float p_env = clamp(params.env_split[0], 0.01f, 0.99f);
    const float flux = params.cdf[params.triangle_count - 1];
    return flux > 0.f ? 4.f * MERIAN_PI * MERIAN_PI * params.env_importance.get_average() *
                            (1.f - p_env) / (p_env * flux)
                      : 1.f;
}

LightSource listed_light(const LightPreprocessBinding params, const uint source) {
    if (!is_env_region(source))
        return params.sources[source];
    const uint region = source - LIGHT_ENV_REGION_BASE;
    LightSource light;
    params.env_importance.region_cone(region, light.centroid, light.radius);
    light.flux = MERIAN_PI * params.env_importance.region_value(region) * MERIAN_FOUR_PI /
                 float(ENV_REGIONS);
    light.normal = float3(0);
    light.flags = uint16_t(EmissiveTriangleFlags::EmissiveDistant);
    light.triangle_count = uint16_t(0);
    return light;
}

// Importance over the pool's density: the flux cancels.
float candidate_weight(const LightPreprocessBinding params,
                       const uint source,
                       const float3 pos,
                       const float radius) {
    if (source == LIGHT_INVALID_INDEX)
        return 0.f;
    if (is_env_region(source))
        return params.env_importance.region_value(source - LIGHT_ENV_REGION_BASE) > 0.f
                   ? env_region_weight(params)
                   : 0.f;
    const LightSource light = params.sources[source];
    return light.is_valid() ? light.geometry_term(pos, float3(0), radius) : 0.f;
}

void close_source(const LightPreprocessBinding params,
                  const uint first,
                  const uint count,
                  const float3 weighted_centroid,
                  const float3 lo,
                  const float3 hi,
                  const float flux,
                  const float3 normal,
                  const float area,
                  const uint flags) {
    LightSource source;
    source.centroid = flux > 0.f ? weighted_centroid / flux : 0.5f * (lo + hi);
    source.flux = flux;
    // a fixture whose faces disagree emits every way, so it must not be culled by one of them
    source.normal = length(normal) >= 0.5f * area ? normalize(normal) : float3(0);
    source.radius = length(max(hi - source.centroid, source.centroid - lo));
    source.flags = uint16_t(flags);
    source.triangle_count = uint16_t(count);
    params.sources_out[first] = source;
}

groupshared float3 s_chunk_lo[LIGHT_SOURCE_CHUNK];
groupshared float3 s_chunk_hi[LIGHT_SOURCE_CHUNK];
groupshared float3 s_chunk_centroid[LIGHT_SOURCE_CHUNK];
groupshared float3 s_chunk_normal[LIGHT_SOURCE_CHUNK];
groupshared float s_chunk_flux[LIGHT_SOURCE_CHUNK];
groupshared uint s_chunk_flags[LIGHT_SOURCE_CHUNK];
groupshared uint s_chunk_geometry[LIGHT_SOURCE_CHUNK];
groupshared uint s_chunk_first[LIGHT_SOURCE_CHUNK];
groupshared float s_chunk_running[LIGHT_SOURCE_CHUNK];
groupshared float s_chunk_source_flux[LIGHT_SOURCE_CHUNK];
groupshared uint s_chunk_source_count[LIGHT_SOURCE_CHUNK];

[shader("compute")][numthreads(LIGHT_SOURCE_CHUNK, 1, 1)] void
sources(uint3 group_id : SV_GroupID,
        uint local_id : SV_GroupIndex,
        ParameterBlock<LightPreprocessBinding> params) {
    const uint begin = group_id.x * LIGHT_SOURCE_CHUNK;
    if (begin >= params.triangle_count)
        return;
    const uint count = min(LIGHT_SOURCE_CHUNK, params.triangle_count - begin);

    if (local_id < count) {
        const EmissiveTriangle tri = params.triangles[begin + local_id];
        const float3 scaled_n = tri.get_scaled_normal();
        s_chunk_lo[local_id] = min(tri.p0, min(tri.p1, tri.p2));
        s_chunk_hi[local_id] = max(tri.p0, max(tri.p1, tri.p2));
        s_chunk_centroid[local_id] = (tri.p0 + tri.p1 + tri.p2) * (1.f / 3.f);
        s_chunk_normal[local_id] = scaled_n;
        s_chunk_flux[local_id] = tri.flux;
        s_chunk_flags[local_id] = uint(tri.flags);
        s_chunk_geometry[local_id] = find_light_geometry(
            params.light_geometries, params.light_geometry_count, begin + local_id);
    }
    GroupMemoryBarrierWithGroupSync();

    if (local_id == 0) {
        const LightGridInfo finest = params.grid_info_prev[0];
        const float max_extent = finest.dimension > 0 && finest.inv_cell_size > 0.f
                                     ? params.grid_source_extent / finest.inv_cell_size
                                     : 1e30f;
        uint first = 0;
        float3 lo = float3(1e30f);
        float3 hi = float3(-1e30f);
        float3 weighted_centroid = float3(0);
        float flux = 0.f;
        float3 normal = float3(0);
        float area = 0.f;
        uint flags = 0;
        for (uint i = 0; i < count; i++) {
            const float3 extent = max(hi, s_chunk_hi[i]) - min(lo, s_chunk_lo[i]);
            if (i > first && (s_chunk_geometry[i] != s_chunk_geometry[i - 1] ||
                              max(extent.x, max(extent.y, extent.z)) > max_extent)) {
                close_source(params, begin + first, i - first, weighted_centroid, lo, hi, flux,
                             normal, area, flags);
                s_chunk_source_flux[first] = flux;
                s_chunk_source_count[first] = i - first;
                first = i;
                lo = float3(1e30f);
                hi = float3(-1e30f);
                weighted_centroid = float3(0);
                flux = 0.f;
                normal = float3(0);
                area = 0.f;
                flags = 0;
            }
            lo = min(lo, s_chunk_lo[i]);
            hi = max(hi, s_chunk_hi[i]);
            weighted_centroid += s_chunk_flux[i] * s_chunk_centroid[i];
            flux += s_chunk_flux[i];
            normal += s_chunk_normal[i];
            area += length(s_chunk_normal[i]);
            flags |= s_chunk_flags[i];
            s_chunk_first[i] = first;
            s_chunk_running[i] = flux;
        }
        close_source(params, begin + first, count - first, weighted_centroid, lo, hi, flux, normal,
                     area, flags);
        s_chunk_source_flux[first] = flux;
        s_chunk_source_count[first] = count - first;
    }
    GroupMemoryBarrierWithGroupSync();

    if (local_id >= count)
        return;
    const uint first = s_chunk_first[local_id];
    const float source_flux = s_chunk_source_flux[first];
    const bool last = local_id + 1 == count || s_chunk_first[local_id + 1] != first;
    params.source_of_out[begin + local_id] = begin + first;
    params.source_cdf_out[begin + local_id] =
        last                ? 1.f
        : source_flux > 0.f ? s_chunk_running[local_id] / source_flux
                            : float(local_id + 1 - first) / float(s_chunk_source_count[first]);
}

static const uint LIGHT_GRID_SETUP_GROUP = 256;
groupshared float3 s_setup_min[LIGHT_GRID_SETUP_GROUP];
groupshared float3 s_setup_max[LIGHT_GRID_SETUP_GROUP];
groupshared float s_setup_distance[LIGHT_GRID_SETUP_GROUP];

// Places the cascades around the camera. The outermost one has to reach the lights the camera
// can see, so its half extent comes from the distance from the camera to the farthest light,
// with the spread of the lights as a floor for a camera standing among them. Every finer cascade
// halves the cell size, so a cell covers a roughly constant angle from the camera at any scale.
[shader("compute")][numthreads(LIGHT_GRID_SETUP_GROUP, 1, 1)] void
grid_setup(uint local_id : SV_GroupIndex, ParameterBlock<LightPreprocessBinding> params) {
    const uint dim = params.grid_dimension;
    const uint cascades = max(params.grid_cascades, 1u);
    float cell = params.grid_cell_size;
    if (cell <= 0.f) {
        float3 bmin = float3(1e30f);
        float3 bmax = float3(-1e30f);
        float max_distance = 0.f;
        for (uint i = local_id; i < params.triangle_count; i += LIGHT_GRID_SETUP_GROUP) {
            const EmissiveLightProxy light = params.proxies[i];
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
        cell =
            reach > 0.f ? max(2.f * reach / (float(dim) * exp2(float(cascades - 1u))), 1e-4f) : 1.f;
        // snapped so the cells keep their world positions as the bounds shift
        cell = exp2(ceil(log2(cell)));
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

// Where a cell sits and the sphere it stands for.
struct CellGeometry {
    float3 center;
    float radius;
    uint cascade;
    uint3 coord;
};

CellGeometry cell_geometry(const RWStructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info,
                           const uint cell,
                           const float jitter) {
    const uint dim = grid_info[0].dimension;
    const uint cells_per_cascade = dim * dim * dim;
    const uint local = cell % cells_per_cascade;

    CellGeometry cg;
    cg.cascade = cell / cells_per_cascade;
    cg.coord = uint3(local % dim, (local / dim) % dim, local / (dim * dim));
    const LightGridInfo info = grid_info[cg.cascade];
    const float cell_size = 1.f / info.inv_cell_size;
    cg.center = info.origin + (float3(cg.coord) + 0.5f) * cell_size;
    // the sphere the cell stands for: its circumradius, grown by the jitter the lookup adds
    cg.radius = 0.5f * sqrt(3.f) * cell_size * (jitter + 1.f);
    return cg;
}

// Independent draws from the flux distribution, so a sample is one load and its density is still
// flux[i] / total.
[shader("compute")][numthreads(64, 1, 1)] void pool(uint3 thread_id : SV_DispatchThreadID,
                                                    ParameterBlock<LightPreprocessBinding> params) {
    const uint i = thread_id.x;
    if (i >= params.pool_size)
        return;
    if (params.triangle_count == 0 || !(params.cdf[params.triangle_count - 1] > 0.f)) {
        params.pool[i] = LIGHT_INVALID_INDEX;
        if (i < LIGHT_GRID_POOL)
            params.pool_source_out[i] = LIGHT_INVALID_INDEX;
        return;
    }

    RandomGenerator rng = PCG(xxhash32(uint2(i, params.frame)));
    const uint index =
        search_flux(params.cdf, rng.next_float() * params.cdf[params.triangle_count - 1], 0,
                    params.triangle_count - 1);
    params.pool[i] = index;

    if (i >= min(params.pool_size, LIGHT_GRID_POOL))
        return;

    params.pool_source_out[i] = params.source_of[index];
}

uint remap_carried(const LightPreprocessBinding params, const uint index) {
    if (is_env_region(index))
        return params.env_regions &&
                       params.env_importance.region_value(index - LIGHT_ENV_REGION_BASE) > 0.f
                   ? index
                   : LIGHT_INVALID_INDEX;
    uint triangle = index;
    if (params.light_remap_count > 0) {
        if (index == LIGHT_INVALID_INDEX)
            return LIGHT_INVALID_INDEX;
        uint lo = 0;
        uint hi = params.light_remap_count - 1;
        while (lo < hi) {
            const uint mid = (lo + hi + 1) >> 1;
            if (params.light_remap[mid].old_first <= index)
                lo = mid;
            else
                hi = mid - 1;
        }
        const LightRemap remap = params.light_remap[lo];
        const uint offset = index - remap.old_first;
        if (index < remap.old_first || offset >= min(remap.old_count, remap.new_count) ||
            remap.new_first == LIGHT_INVALID_INDEX)
            return LIGHT_INVALID_INDEX;
        triangle = remap.new_first + offset;
    }
    if (triangle >= params.triangle_count)
        return LIGHT_INVALID_INDEX;
    const uint source = params.source_of[triangle];
    return params.sources[source].is_valid() ? source : LIGHT_INVALID_INDEX;
}

// Threads that build one cell.
static const uint LIGHT_GRID_BUILD_GROUP = 64;

// Neighbouring cells a cell takes a find from each frame.
static const uint LIGHT_GRID_NEIGHBOURS = 3;

static const uint LIGHT_GRID_CANDIDATE_ENTRIES =
    LIGHT_GRID_MAX_NEW_LIGHTS + LIGHT_GRID_NEIGHBOURS + LIGHT_GRID_DISCOVERED;
groupshared uint s_candidate[LIGHT_GRID_CANDIDATE_ENTRIES];
groupshared float s_candidate_key[LIGHT_GRID_CANDIDATE_ENTRIES];
groupshared float s_candidate_visibility[LIGHT_GRID_CANDIDATE_ENTRIES];
groupshared uint s_slot_index[LIGHT_GRID_SLOTS];
groupshared float s_slot_key[LIGHT_GRID_SLOTS];
groupshared float s_slot_visibility[LIGHT_GRID_SLOTS];

bool reproject_cell(const StructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info_prev,
                    const LightGridInfo info,
                    const uint3 coord,
                    out uint prev_cascade,
                    out uint3 prev_coord) {
    prev_cascade = 0;
    prev_coord = coord;
    const LightGridInfo finest = grid_info_prev[0];
    if (finest.dimension != info.dimension || !(finest.inv_cell_size > 0.f))
        return false;
    const int cascade = int(round(log2(finest.inv_cell_size / info.inv_cell_size)));
    if (cascade < 0 || cascade >= int(finest.cascade_count))
        return false;
    const LightGridInfo prev = grid_info_prev[cascade];
    if (abs(prev.inv_cell_size - info.inv_cell_size) > 1e-6f * info.inv_cell_size)
        return false;
    const int3 delta = int3(round((info.origin - prev.origin) * info.inv_cell_size));
    const int3 c = int3(coord) + delta;
    if (any(c < 0) || any(c >= int(info.dimension)))
        return false;
    prev_cascade = uint(cascade);
    prev_coord = uint3(c);
    return true;
}

/* Each cell keeps the lights that reach it, ranked by what they deliver there.
 *
 * A candidate's key is the reciprocal of its importance for the cell and the cell keeps the
 * LIGHT_GRID_SLOTS smallest. Keys carry across frames, so the list ranks every candidate the cell
 * has ever seen rather than this frame's; they also age, so it follows the scene when the lights
 * or the camera move. One slot per source keeps the list on distinct lights.
 */
[shader("compute")][numthreads(LIGHT_GRID_BUILD_GROUP, 1, 1)] void
grid(uint3 group_id : SV_GroupID,
     uint local_id : SV_GroupIndex,
     ParameterBlock<LightPreprocessBinding> params) {
    const uint dim = params.grid_info[0].dimension;
    const uint cells_per_cascade = dim * dim * dim;
    const uint cell = group_id.x + group_id.y * LIGHT_GRID_DISPATCH_ROW;
    if (cell >= cells_per_cascade * params.grid_info[0].cascade_count)
        return;

    const uint seed = xxhash32(uint3(cell, params.frame, 0x9e3779b9u));
    const CellGeometry cg = cell_geometry(params.grid_info, cell, params.grid_jitter);
    const uint cascade = cg.cascade;
    const uint3 coord = cg.coord;
    const float3 center = cg.center;
    const float radius = cg.radius;
    const uint new_lights = clamp(params.grid_new_lights, 1u, LIGHT_GRID_MAX_NEW_LIGHTS);
    const uint grid_pool = max(min(params.pool_size, LIGHT_GRID_POOL), 1u);
    // a key lengthens by this much a frame, so a stale one is overtaken after grid_max_age
    const float ageing = 1.f + 1.f / float(max(params.grid_max_age, 1u));

    uint prev_cascade;
    uint3 prev_coord;
    const bool carry = !params.grid_reset && reproject_cell(params.grid_info_prev,
                                                            params.grid_info[cascade], coord,
                                                            prev_cascade, prev_coord);

    const uint neighbours = new_lights;
    const uint discovered = neighbours + LIGHT_GRID_NEIGHBOURS;
    const uint ranked = discovered + LIGHT_GRID_DISCOVERED;
    const uint prev_cell = prev_cascade * cells_per_cascade + prev_coord.x +
                           dim * (prev_coord.y + dim * prev_coord.z);

    // 1. carried slots and new lights
    if (local_id < LIGHT_GRID_SLOTS) {
        uint held = LIGHT_INVALID_INDEX;
        float key = 1e30f;
        float visibility = LIGHT_GRID_PRIOR_VISIBILITY;
        if (carry) {
            const uint entry = prev_cell * LIGHT_GRID_SLOTS + local_id;
            held = remap_carried(params, params.grid_prev[entry]);
            if (held != LIGHT_INVALID_INDEX) {
                const float prev_visibility = params.grid_visibility_prev[entry];
                const float tries = float(params.grid_feedback_prev[2 * entry]);
                const float hits = float(params.grid_feedback_prev[2 * entry + 1]);
                visibility = (hits + LIGHT_GRID_PRIOR_SAMPLES * prev_visibility) /
                             (tries + LIGHT_GRID_PRIOR_SAMPLES);
                key = params.grid_key_prev[entry] * ageing *
                      max(prev_visibility, LIGHT_GRID_MIN_VISIBILITY) /
                      max(visibility, LIGHT_GRID_MIN_VISIBILITY);
            }
        }
        s_slot_index[local_id] = held;
        s_slot_key[local_id] = key;
        s_slot_visibility[local_id] = visibility;
    }

    for (uint c = local_id; c < new_lights; c += LIGHT_GRID_BUILD_GROUP) {
        RandomGenerator rng = PCG(xxhash32(uint2(seed, c)));
        const uint pooled =
            params.pool_source[min(uint(rng.next_float() * grid_pool), grid_pool - 1)];
        const uint source = pooled < params.triangle_count ? pooled : LIGHT_INVALID_INDEX;
        s_candidate[c] = source;
        s_candidate_key[c] = candidate_weight(params, source, center, radius);
        s_candidate_visibility[c] = LIGHT_GRID_PRIOR_VISIBILITY;
    }

    // neighbours' slots
    if (local_id >= LIGHT_GRID_BUILD_GROUP - LIGHT_GRID_NEIGHBOURS) {
        const uint n_id = LIGHT_GRID_BUILD_GROUP - 1u - local_id;
        uint index = LIGHT_INVALID_INDEX;
        float weight = 0.f;
        float visibility = LIGHT_GRID_PRIOR_VISIBILITY;
        if (carry) {
            RandomGenerator rng = PCG(xxhash32(uint2(seed ^ 0x9e3779b9u, n_id)));
            const uint axis = min(uint(rng.next_float() * 3.f), 2u);
            int3 n = int3(prev_coord);
            n[axis] += select(rng.next_float() < 0.5f, -1, 1);
            if (all(n >= 0) && all(n < int(dim))) {
                const uint neighbour = prev_cascade * cells_per_cascade + uint(n.x) +
                                       dim * (uint(n.y) + dim * uint(n.z));
                const uint slot =
                    min(uint(rng.next_float() * LIGHT_GRID_SLOTS), LIGHT_GRID_SLOTS - 1);
                const uint neighbour_entry = neighbour * LIGHT_GRID_SLOTS + slot;
                index = remap_carried(params, params.grid_prev[neighbour_entry]);
                if (index != LIGHT_INVALID_INDEX) {
                    visibility = params.grid_visibility_prev[neighbour_entry];
                    // judged for this cell, not the one it came from
                    weight = candidate_weight(params, index, center, radius);
                }
            }
        }
        s_candidate[neighbours + n_id] = index;
        s_candidate_key[neighbours + n_id] = weight;
        s_candidate_visibility[neighbours + n_id] = visibility;
    }

    if (local_id < LIGHT_GRID_DISCOVERED) {
        uint index = LIGHT_INVALID_INDEX;
        float weight = 0.f;
        if (carry) {
            index = remap_carried(
                params, params.grid_discovered_prev[prev_cell * LIGHT_GRID_DISCOVERED + local_id]);
            weight = candidate_weight(params, index, center, radius);
        }
        s_candidate[discovered + local_id] = index;
        s_candidate_key[discovered + local_id] = weight;
        s_candidate_visibility[discovered + local_id] = 1.f;
        params.grid_discovered[cell * LIGHT_GRID_DISCOVERED + local_id] = LIGHT_INVALID_INDEX;
    }
    GroupMemoryBarrierWithGroupSync();

    // 2. keys
    for (uint c = local_id; c < ranked; c += LIGHT_GRID_BUILD_GROUP) {
        const uint index = s_candidate[c];
        const float weight = s_candidate_key[c];
        uint held = LIGHT_GRID_SLOTS;
        for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
            if (s_slot_index[slot] == index) {
                held = slot;
                break;
            }
        }
        const float visibility =
            held < LIGHT_GRID_SLOTS ? s_slot_visibility[held] : s_candidate_visibility[c];
        float key = 1e30f;
        if (index != LIGHT_INVALID_INDEX && weight > 0.f) {
            // without replacement, so a cell that sees more sources than it has slots does not
            // collapse onto the largest (A-Res, Efraimidis and Spirakis 2006)
            RandomGenerator key_rng = PCG(xxhash32(uint3(seed, c, 0x85ebca6bu)));
            key = (params.grid_a_res ? -log(max(key_rng.next_float(), 1e-7f)) / weight
                                     : 1.f / weight) /
                  max(visibility, LIGHT_GRID_MIN_VISIBILITY);
        }
        s_candidate_key[c] = key;
        s_candidate_visibility[c] = visibility;
    }

    bool carried_twice = false;
    if (local_id < LIGHT_GRID_SLOTS && s_slot_index[local_id] != LIGHT_INVALID_INDEX) {
        for (uint slot = 0; slot < local_id; slot++) {
            carried_twice |= s_slot_index[slot] == s_slot_index[local_id];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (carried_twice) {
        s_slot_index[local_id] = LIGHT_INVALID_INDEX;
        s_slot_key[local_id] = 1e30f;
        s_slot_visibility[local_id] = LIGHT_GRID_PRIOR_VISIBILITY;
    }
    GroupMemoryBarrierWithGroupSync();

    // 3. rank against the slots, a slot per lane of the first wave
    if (local_id < WaveGetLaneCount()) {
        const bool has_slot = local_id < LIGHT_GRID_SLOTS;
        uint slot_index = has_slot ? s_slot_index[local_id] : LIGHT_INVALID_INDEX;
        float slot_key = has_slot ? s_slot_key[local_id] : -1.f;
        float slot_visibility = has_slot ? s_slot_visibility[local_id] : 0.f;
        float worst_key = WaveActiveMax(slot_key);
        for (uint c = 0; c < ranked; c++) {
            float key = s_candidate_key[c];
            if (key >= worst_key)
                continue;
            const uint index = s_candidate[c];
            float visibility = s_candidate_visibility[c];
            const uint held = WaveActiveBallot(slot_index == index).x;
            uint target;
            if (held != 0) {
                target = firstbitlow(held);
                const float held_visibility = WaveReadLaneAt(slot_visibility, target);
                key *= max(visibility, LIGHT_GRID_MIN_VISIBILITY) /
                       max(held_visibility, LIGHT_GRID_MIN_VISIBILITY);
                visibility = held_visibility;
            } else {
                target = firstbitlow(WaveActiveBallot(slot_key == worst_key).x);
            }
            if (WaveGetLaneIndex() == target && key < slot_key) {
                slot_index = index;
                slot_key = key;
                slot_visibility = visibility;
            }
            worst_key = WaveActiveMax(slot_key);
        }
        if (has_slot) {
            s_slot_index[local_id] = slot_index;
            s_slot_key[local_id] = slot_key;
            s_slot_visibility[local_id] = slot_visibility;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (local_id >= LIGHT_GRID_SLOTS)
        return;

    const uint entry = cell * LIGHT_GRID_SLOTS + local_id;
    const uint held = s_slot_index[local_id];
    params.grid[entry] = held;
    params.grid_key[entry] = s_slot_key[local_id];
    params.grid_visibility[entry] = s_slot_visibility[local_id];
    const bool listed = held != LIGHT_INVALID_INDEX;
    LightSource slot_source = {};
    if (listed) {
        slot_source = listed_light(params, held);
        slot_source.flux *= max(s_slot_visibility[local_id], LIGHT_GRID_MIN_DRAW_VISIBILITY);
    }
    if (params.slot_weighing == LightSlotWeighing::LightSlotWeighingShadingPoint) {
        params.grid_slot_sources[entry] = slot_source;
    } else {
        // the slots are the first lanes of one wave
        const float weight =
            listed ? slot_source.flux * slot_source.geometry_term(center, float3(0), radius) : 0.f;
        const float total = WaveActiveSum(weight);
        const float even = total > 0.f ? params.grid_even_share : 1.f;
        params.grid_probability[entry] =
            listed ? (1.f - even) * (total > 0.f ? weight / total : 0.f) +
                         even / float(WaveActiveCountBits(listed))
                   : 0.f;
    }
    params.grid_feedback[2 * entry] = 0;
    params.grid_feedback[2 * entry + 1] = 0;
}
