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
    // where the entries the grid was ranked against sit now; empty while the table is unchanged
    StructuredBuffer<uint> light_remap;
    uint light_remap_count;
    RWStructuredBuffer<EmissiveLightProxy, ScalarDataLayout> geometry_proxies_out;
    // light source each pool entry landed in
    RWStructuredBuffer<uint> pool_geometry_out;
    StructuredBuffer<uint> pool_geometry;
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> geometry_proxies;
    uint light_geometry_count;
    RWStructuredBuffer<uint> grid;
    RWStructuredBuffer<float> grid_weight;
    StructuredBuffer<uint> grid_prev;
    StructuredBuffer<float> grid_weight_prev;
    RWStructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info;
    StructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info_prev;

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
    // frames over which a key that stopped earning its slot is overtaken
    uint grid_max_age;
    // drop everything carried over, for a frame the previous grid cannot describe
    uint grid_reset;
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

// Importance of a light for a point over the density the pool draws it with: the pool is already
// proportional to the flux, so what remains is the geometry term towards the point.
float resampling_weight(const EmissiveLightProxy light, const float3 pos, const float radius) {
    if (light.flux <= 0.f)
        return 0.f;
    const float3 to_pos = pos - light.centroid;
    const float area2 = length(light.scaled_normal);
    if (area2 <= 0.f)
        return 0.f;
    const float dist2 = max(dot(to_pos, to_pos), 1e-12f);
    const float dist = sqrt(dist2);
    float cos_theta = dot(light.scaled_normal / area2, to_pos) / dist;
    if (light.is_two_sided())
        cos_theta = abs(cos_theta);
    // the emitter's cone widened by the angle the cell subtends, so the weight survives while any
    // point of the cell is still in front of it (Conty Estevez and Kulla 2018)
    const float sin_u = min(radius / dist, 1.f);
    const float cos_u = sqrt(max(1.f - sin_u * sin_u, 0.f));
    const float sin_theta = sqrt(max(1.f - cos_theta * cos_theta, 0.f));
    const float cos_l = cos_theta >= cos_u ? 1.f : cos_theta * cos_u + sin_theta * sin_u;
    if (cos_l <= 0.f)
        return 0.f;
    // clamped by the emitter's own size, so a point on it is not a singularity
    return cos_l / max(dist2, 0.5f * area2);
}

static const uint LIGHT_GEOMETRY_GROUP = 64;
groupshared float3 s_geometry_centroid[LIGHT_GEOMETRY_GROUP];
groupshared float3 s_geometry_normal[LIGHT_GEOMETRY_GROUP];
groupshared float2 s_geometry_flux[LIGHT_GEOMETRY_GROUP];
groupshared uint s_geometry_flags[LIGHT_GEOMETRY_GROUP];

// One proxy per emissive geometry, so a cell can hold light sources rather than triangles.
[shader("compute")][numthreads(LIGHT_GEOMETRY_GROUP, 1, 1)] void
geometry_proxy(uint3 group_id : SV_GroupID,
               uint local_id : SV_GroupIndex,
               ParameterBlock<LightPreprocessBinding> params) {
    const uint g = group_id.x;
    if (g >= params.light_geometry_count)
        return;
    const LightGeometry geometry = params.light_geometries[g];

    float3 centroid = float3(0);
    float3 normal = float3(0);
    float flux = 0.f;
    float area = 0.f;
    uint flags = 0;
    for (uint i = local_id; i < geometry.primitive_count; i += LIGHT_GEOMETRY_GROUP) {
        const EmissiveLightProxy proxy = params.proxies[geometry.first_triangle + i];
        flux += proxy.flux;
        centroid += proxy.centroid * proxy.flux;
        normal += proxy.scaled_normal;
        area += length(proxy.scaled_normal);
        flags |= proxy.flags;
    }
    s_geometry_centroid[local_id] = centroid;
    s_geometry_normal[local_id] = normal;
    s_geometry_flux[local_id] = float2(flux, area);
    s_geometry_flags[local_id] = flags;
    GroupMemoryBarrierWithGroupSync();

    if (local_id != 0)
        return;
    for (uint i = 1; i < LIGHT_GEOMETRY_GROUP; i++) {
        centroid += s_geometry_centroid[i];
        normal += s_geometry_normal[i];
        flux += s_geometry_flux[i].x;
        area += s_geometry_flux[i].y;
        flags |= s_geometry_flags[i];
    }

    EmissiveLightProxy proxy;
    proxy.centroid = select(flux > 0.f, centroid / max(flux, 1e-30f), float3(0));
    proxy.flux = flux;
    proxy.scaled_normal = normal;
    // a fixture whose faces disagree emits every way, so it must not be culled by one of them
    proxy.flags = flags;
    if (length(normal) < 0.5f * area)
        proxy.flags |= uint(EmissiveTriangleFlags::EmissiveTwoSided);
    params.geometry_proxies_out[g] = proxy;
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
        for (uint i = local_id; i < params.light_geometry_count; i += LIGHT_GRID_SETUP_GROUP) {
            const EmissiveLightProxy light = params.geometry_proxies[i];
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
        params.pool_geometry_out[i] = LIGHT_INVALID_INDEX;
        return;
    }

    RandomGenerator rng = PCG(xxhash32(uint2(i, params.frame)));
    const uint index =
        search_flux(params.cdf, rng.next_float() * params.cdf[params.triangle_count - 1], 0,
                    params.triangle_count - 1);
    params.pool[i] = index;

    if (i >= min(params.pool_size, LIGHT_GRID_POOL))
        return;

    params.pool_geometry_out[i] =
        find_light_geometry(params.light_geometries, params.light_geometry_count, index);
}

// Where an entry the grid was ranked against sits in this frame's table.
uint remap_carried(const LightPreprocessBinding params, const uint index) {
    const uint mapped =
        params.light_remap_count == 0
            ? index
            : (index < params.light_remap_count ? params.light_remap[index] : LIGHT_INVALID_INDEX);
    return mapped < params.light_geometry_count ? mapped : LIGHT_INVALID_INDEX;
}

// Threads that build one cell.
static const uint LIGHT_GRID_BUILD_GROUP = 64;

// Neighbouring cells a cell takes a find from each frame.
static const uint LIGHT_GRID_NEIGHBOURS = 3;

// this frame's candidates and their weight for the cell
groupshared uint s_index[LIGHT_GRID_MAX_CANDIDATES];
groupshared float s_weight[LIGHT_GRID_MAX_CANDIDATES];
// what the neighbours pass on this frame
groupshared uint s_neighbour[LIGHT_GRID_NEIGHBOURS];
groupshared float s_neighbour_weight[LIGHT_GRID_NEIGHBOURS];
// the cell's sample and the key each slot won with
groupshared uint s_slot_index[LIGHT_GRID_SLOTS];
groupshared float s_slot_draw[LIGHT_GRID_SLOTS];

// Where this cell sat in the previous frame's grid. The cascade origins are snapped to their own
// cell size, so a camera move is a whole-cell shift and the lookup is an integer offset.
bool reproject_cell(const LightGridInfo info,
                    const LightGridInfo prev,
                    const uint3 coord,
                    out uint3 prev_coord) {
    prev_coord = coord;
    if (prev.dimension != info.dimension || prev.cascade_count != info.cascade_count)
        return false;
    if (abs(prev.inv_cell_size - info.inv_cell_size) > 1e-6f * info.inv_cell_size)
        return false;
    const int3 delta = int3(round((info.origin - prev.origin) * info.inv_cell_size));
    const int3 c = int3(coord) + delta;
    if (any(c < 0) || any(c >= int(info.dimension)))
        return false;
    prev_coord = uint3(c);
    return true;
}

/* Each cell keeps the lights that reach it, ranked by what they deliver there.
 *
 * A candidate's key is the reciprocal of its importance for the cell and the cell keeps the
 * LIGHT_GRID_SLOTS smallest. Keys carry across frames, so the list ranks every candidate the cell
 * has ever seen rather than this frame's; they also age, so it follows the scene when the lights
 * or the camera move. One slot per source keeps the list on distinct lights.
 *
 */
[shader("compute")][numthreads(LIGHT_GRID_BUILD_GROUP, 1, 1)] void
grid(uint3 group_id : SV_GroupID,
     uint local_id : SV_GroupIndex,
     ParameterBlock<LightPreprocessBinding> params) {
    const uint dim = params.grid_info[0].dimension;
    const uint cells_per_cascade = dim * dim * dim;
    const uint cell = group_id.x;
    if (dim == 0 || cell >= cells_per_cascade * params.grid_info[0].cascade_count)
        return;

    const uint seed = xxhash32(uint3(cell, params.frame, 0x9e3779b9u));
    const CellGeometry cg = cell_geometry(params.grid_info, cell, params.grid_jitter);
    const uint cascade = cg.cascade;
    const uint3 coord = cg.coord;
    const float3 center = cg.center;
    const float radius = cg.radius;
    const uint candidates = clamp(params.grid_candidates, 1u, LIGHT_GRID_MAX_CANDIDATES);
    const uint grid_pool = max(min(params.pool_size, LIGHT_GRID_POOL), 1u);
    // a key lengthens by this much a frame, so a stale one is overtaken after grid_max_age
    const float ageing = 1.f + 1.f / float(max(params.grid_max_age, 1u));

    uint3 prev_coord;
    const bool carry =
        params.grid_reset == 0 && reproject_cell(params.grid_info[cascade],
                                                 params.grid_info_prev[cascade], coord, prev_coord);

    // 1. this frame's candidates, weighed for this cell
    const uint chunk = (candidates + LIGHT_GRID_BUILD_GROUP - 1u) / LIGHT_GRID_BUILD_GROUP;
    const uint first = local_id * chunk;

    for (uint k = 0; k < chunk; k++) {
        const uint c = first + k;
        if (c >= candidates)
            break;
        RandomGenerator rng = PCG(xxhash32(uint2(seed, c)));
        const uint entry = min(uint(rng.next_float() * grid_pool), grid_pool - 1);
        const uint geometry = params.pool_geometry[entry];
        // the pool already draws by flux, so what remains to weigh is the geometry term
        s_index[c] = geometry < params.light_geometry_count ? geometry : LIGHT_INVALID_INDEX;
        s_weight[c] = geometry < params.light_geometry_count
                          ? resampling_weight(params.geometry_proxies[geometry], center, radius)
                          : 0.f;
    }

    if (local_id < LIGHT_GRID_NEIGHBOURS) {
        s_neighbour[local_id] = LIGHT_INVALID_INDEX;
        s_neighbour_weight[local_id] = 0.f;
    }
    GroupMemoryBarrierWithGroupSync();

    // 2. what the neighbours found: the pool rarely offers a light that dominates one corner
    if (local_id >= LIGHT_GRID_BUILD_GROUP - LIGHT_GRID_NEIGHBOURS && carry) {
        const uint n_id = LIGHT_GRID_BUILD_GROUP - 1u - local_id;
        RandomGenerator rng = PCG(xxhash32(uint2(seed ^ 0x9e3779b9u, n_id)));
        const uint axis = min(uint(rng.next_float() * 3.f), 2u);
        int3 n = int3(prev_coord);
        n[axis] += select(rng.next_float() < 0.5f, -1, 1);
        if (all(n >= 0) && all(n < int(dim))) {
            const uint neighbour =
                cascade * cells_per_cascade + uint(n.x) + dim * (uint(n.y) + dim * uint(n.z));
            const uint slot = min(uint(rng.next_float() * LIGHT_GRID_SLOTS), LIGHT_GRID_SLOTS - 1);
            const uint index =
                remap_carried(params, params.grid_prev[neighbour * LIGHT_GRID_SLOTS + slot]);
            if (index != LIGHT_INVALID_INDEX) {
                s_neighbour[n_id] = index;
                // judged for this cell, not the one it came from
                s_neighbour_weight[n_id] =
                    resampling_weight(params.geometry_proxies[index], center, radius);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // 3. the ranking: what the cell carries, then this frame's candidates against it
    if (local_id < LIGHT_GRID_SLOTS) {
        uint held = LIGHT_INVALID_INDEX;
        float draw = 1e30f;
        if (carry) {
            const uint prev_cell = cascade * cells_per_cascade + prev_coord.x +
                                   dim * (prev_coord.y + dim * prev_coord.z);
            const uint entry = prev_cell * LIGHT_GRID_SLOTS + local_id;
            held = remap_carried(params, params.grid_prev[entry]);
            if (held != LIGHT_INVALID_INDEX) {
                draw = params.grid_weight_prev[entry] * ageing;
            }
        }
        s_slot_index[local_id] = held;
        s_slot_draw[local_id] = draw;
    }
    GroupMemoryBarrierWithGroupSync();

    if (local_id == 0) {
        uint worst = 0;
        float worst_draw = 0.f;
        for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
            if (s_slot_draw[slot] > worst_draw) {
                worst = slot;
                worst_draw = s_slot_draw[slot];
            }
        }
        for (uint c = 0; c < candidates + LIGHT_GRID_NEIGHBOURS; c++) {
            uint index;
            float weight;
            if (c < candidates) {
                index = s_index[c];
                weight = s_weight[c];
            } else {
                index = s_neighbour[c - candidates];
                weight = s_neighbour_weight[c - candidates];
            }
            if (index == LIGHT_INVALID_INDEX || !(weight > 0.f))
                continue;
            const float draw = 1.f / weight;

            if (draw >= worst_draw)
                continue;

            uint duplicate = LIGHT_GRID_SLOTS;
            for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
                if (s_slot_index[slot] == index) {
                    duplicate = slot;
                    break;
                }
            }
            // one slot per source, so a light already held only improves its key
            const uint target = duplicate < LIGHT_GRID_SLOTS ? duplicate : worst;
            if (draw < s_slot_draw[target]) {
                s_slot_index[target] = index;
                s_slot_draw[target] = draw;
                worst = 0;
                worst_draw = 0.f;
                for (uint slot = 0; slot < LIGHT_GRID_SLOTS; slot++) {
                    if (s_slot_draw[slot] > worst_draw) {
                        worst = slot;
                        worst_draw = s_slot_draw[slot];
                    }
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (local_id >= LIGHT_GRID_SLOTS)
        return;

    const uint entry = cell * LIGHT_GRID_SLOTS + local_id;
    params.grid[entry] = s_slot_index[local_id];
    params.grid_weight[entry] = s_slot_draw[local_id];
}
