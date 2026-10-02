#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.scene.env_importance;
import merian_shaders.scene.light_data;
import merian_shaders.utils.envmap;
import merian_shaders.utils.hash;
import merian_shaders.utils.morton;
import merian_shaders.utils.random;

using merian;

// Per-frame preprocess for next event estimation: the share the environment takes of the light
// samples, a pool drawn from the flux distribution, and a world-space grid of per-cell cuts through
// the light tree.

struct LightPreprocessBinding {
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    StructuredBuffer<uint> tree_keys;
    StructuredBuffer<uint> tree_order;
    StructuredBuffer<float2> tree_cdf;
    StructuredBuffer<float2> env_cdf;
    // 0 where the cells do not list the environment
    uint env_size;
    RWStructuredBuffer<LightTreeInfo, ScalarDataLayout> tree_info;
    StructuredBuffer<LightTreeInfo, ScalarDataLayout> tree_info_prev;
    RWStructuredBuffer<uint> pool;
    RWStructuredBuffer<uint> grid_keys;
    StructuredBuffer<uint> grid_keys_prev;
    RWStructuredBuffer<float4> grid_contribution;
    StructuredBuffer<float4> grid_contribution_prev;
    RWStructuredBuffer<float> grid_estimate;
    RWStructuredBuffer<uint> grid_feedback;
    StructuredBuffer<uint> grid_feedback_prev;
    RWStructuredBuffer<uint> grid_starts;
    RWStructuredBuffer<float> grid_probability;
    RWStructuredBuffer<LightBound, ScalarDataLayout> grid_slot_bounds;
    RWStructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info;
    StructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info_prev;

    // fraction of the light bounds the grid spans around the camera
    float grid_coverage;
    float3 camera_position;
    uint triangle_count;
    uint pool_size;
    uint grid_dimension;
    uint grid_cascades;
    float grid_cell_size;
    float grid_jitter;
    float grid_even_share;
    uint grid_refinements;
    LightSlotWeighing slot_weighing;
    // drop everything carried over, for a frame the previous grid cannot describe
    bool grid_reset;
    uint frame;
};

struct EnvSplitBinding {
    // inclusive prefix sum of the triangle flux
    StructuredBuffer<float2> cdf;
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
    const float2 total =
        params.triangle_count > 0 ? params.cdf[params.triangle_count - 1] : float2(0);
    const float triangles = max(total.x + total.y, 0.f);
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

static const uint LIGHT_GRID_SETUP_GROUP = 256;
groupshared float3 s_setup_min[LIGHT_GRID_SETUP_GROUP];
groupshared float3 s_setup_max[LIGHT_GRID_SETUP_GROUP];
groupshared float s_setup_distance[LIGHT_GRID_SETUP_GROUP];

bool same_lattice(const LightTreeInfo a, const LightTreeInfo b) {
    return all(asuint(a.origin) == asuint(b.origin)) &&
           asuint(a.inv_cell_size) == asuint(b.inv_cell_size);
}

// Fits the light tree's lattice to the lights and places the cascades around the camera. The
// outermost one has to reach the lights the camera can see, so its half extent comes from the
// distance from the camera to the farthest light, with the spread of the lights as a floor for a
// camera standing among them. Every finer cascade halves the cell size, so a cell covers a roughly
// constant angle from the camera at any scale.
[shader("compute")][numthreads(LIGHT_GRID_SETUP_GROUP, 1, 1)] void
grid_setup(uint local_id : SV_GroupIndex, ParameterBlock<LightPreprocessBinding> params) {
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

    if (local_id == 0) {
        LightTreeInfo info = params.tree_info_prev[0];
        if (bmin.x <= bmax.x) {
            const float lattice_cells = float(1u << LIGHT_TREE_SIDE_BITS);
            const float3 spread = bmax - bmin;
            const float needed =
                exp2(ceil(log2(max(2.f * max(spread.x, max(spread.y, spread.z)), 1e-3f))));
            const float side = info.inv_cell_size > 0.f ? lattice_cells / info.inv_cell_size : 0.f;
            const bool holds = side > 0.f && side <= 2.f * needed && all(bmin >= info.origin) &&
                               all(bmax < info.origin + side);
            if (!holds) {
                const float quarter = 0.25f * needed;
                info.origin = floor(bmin / quarter) * quarter;
                info.inv_cell_size = lattice_cells / needed;
            }
        } else if (!(info.inv_cell_size > 0.f)) {
            info.origin = float3(0);
            info.inv_cell_size = 1.f;
        }
        params.tree_info[0] = info;
    }

    const uint dim = params.grid_dimension;
    const uint cascades = max(params.grid_cascades, 1u);
    float cell = params.grid_cell_size;
    if (cell <= 0.f) {
        const float extent = length(max(bmax - bmin, float3(0)));
        const float reach = params.grid_coverage * max(max_distance, 0.5f * extent);
        // the coarsest cascade spans 2 * reach, every finer one half of the one before
        cell =
            reach > 0.f ? max(2.f * reach / (float(dim) * exp2(float(cascades - 1u))), 1e-4f) : 1.f;
        // snapped so the cells keep their world positions as the bounds shift
        cell = exp2(ceil(log2(cell)));
        const LightGridInfo prev = params.grid_info_prev[0];
        if (!params.grid_reset && prev.dimension == dim && prev.cascade_count == cascades &&
            prev.inv_cell_size > 0.f) {
            const float prev_cell = 1.f / prev.inv_cell_size;
            if (cell < prev_cell && cell >= 0.5f * prev_cell)
                cell = prev_cell;
        }
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
    const uint count = params.triangle_count;
    const float2 total = count > 0 ? params.tree_cdf[count - 1] : float2(0);
    if (!(total.x + total.y > 0.f)) {
        params.pool[i] = LIGHT_INVALID_INDEX;
        return;
    }
    RandomGenerator rng = PCG(xxhash32(uint2(i, params.frame)));
    params.pool[i] = params.tree_order[search_range(params.tree_cdf, 0, count, rng.next_float())];
}

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

bool coarser_cell(const StructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info_prev,
                  const LightGridInfo info,
                  const float3 pos,
                  out uint prev_cell) {
    prev_cell = 0;
    const LightGridInfo finest = grid_info_prev[0];
    if (finest.dimension != info.dimension || !(finest.inv_cell_size > 0.f))
        return false;
    const int cascade = int(round(log2(finest.inv_cell_size / info.inv_cell_size))) + 1;
    if (cascade < 1 || cascade >= int(finest.cascade_count))
        return false;
    const LightGridInfo prev = grid_info_prev[cascade];
    if (abs(2.f * prev.inv_cell_size - info.inv_cell_size) > 1e-6f * info.inv_cell_size)
        return false;
    const int3 c = int3(floor((pos - prev.origin) * prev.inv_cell_size));
    const uint dim = info.dimension;
    if (any(c < 0) || any(c >= int(dim)))
        return false;
    prev_cell = uint(cascade) * dim * dim * dim + uint(c.x) + dim * (uint(c.y) + dim * uint(c.z));
    return true;
}

uint env_texels(const LightPreprocessBinding params) {
    return params.env_size * params.env_size;
}

uint light_count(const LightPreprocessBinding params) {
    return params.triangle_count + env_texels(params);
}

uint lower_bound(const LightPreprocessBinding params, const uint key, uint lo, uint hi) {
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (params.tree_keys[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

uint node_start(const LightPreprocessBinding params, const uint key) {
    if (key == LIGHT_TREE_NO_KEY)
        return light_count(params);
    if (key >= LIGHT_TREE_ENV_KEY)
        return params.triangle_count + min(key - LIGHT_TREE_ENV_KEY, env_texels(params));
    return lower_bound(params, key, 0, params.triangle_count);
}

struct TreeNode {
    float flux;
    float3 center;
    // half the box's diagonal, or the sine of the cone's half angle
    float radius;
    float spread;
    bool distant;
    bool splittable;
    uint split_key;

    static TreeNode empty() {
        TreeNode node;
        node.flux = 0.f;
        node.center = float3(0);
        node.radius = 0.f;
        node.spread = 0.f;
        node.distant = false;
        node.splittable = false;
        node.split_key = LIGHT_TREE_NO_KEY;
        return node;
    }
};

uint shared_prefix_free_bits(const uint first, const uint last) {
    return first != last ? firstbithigh(first ^ last) + 1u : 0u;
}

TreeNode tree_node(const LightPreprocessBinding params, const uint first, const uint end) {
    TreeNode node = TreeNode::empty();
    if (first >= end)
        return node;
    const uint count = params.triangle_count;
    if (first < count) {
        node.flux = max(
            ff_diff(params.tree_cdf[end - 1], first > 0 ? params.tree_cdf[first - 1] : float2(0)),
            0.f);
        const uint first_key = params.tree_keys[first];
        const uint last_key = params.tree_keys[end - 1];
        const uint free_bits = shared_prefix_free_bits(first_key, last_key);
        node.splittable = free_bits > 0;
        if (node.splittable)
            node.split_key = (last_key >> (free_bits - 1u)) << (free_bits - 1u);
        const uint3 axis_free = uint3((free_bits + 2u) / 3u, (free_bits + 1u) / 3u, free_bits / 3u);
        const uint3 lo = (morton_decode3d(first_key) >> axis_free) << axis_free;
        const float3 size = float3(uint3(1u) << axis_free);
        const LightTreeInfo info = params.tree_info[0];
        node.center = info.origin + (float3(lo) + 0.5f * size) / info.inv_cell_size;
        node.radius = 0.5f * length(size) / info.inv_cell_size;
        return node;
    }

    const uint first_texel = first - count;
    const uint last_texel = end - 1u - count;
    const float texels = float(env_texels(params));
    node.flux = max(ff_diff(params.env_cdf[last_texel],
                            first_texel > 0 ? params.env_cdf[first_texel - 1] : float2(0)),
                    0.f) *
                (MERIAN_PI * MERIAN_FOUR_PI / texels);
    const uint free_bits = shared_prefix_free_bits(first_texel, last_texel);
    node.splittable = free_bits > 0;
    if (node.splittable)
        node.split_key =
            LIGHT_TREE_ENV_KEY + ((last_texel >> (free_bits - 1u)) << (free_bits - 1u));
    const uint2 axis_free = uint2((free_bits + 1u) / 2u, free_bits / 2u);
    const uint2 lo = (morton_decode2d(first_texel) >> axis_free) << axis_free;
    const float2 size = float2(uint2(1u) << axis_free);
    const float inv_side = 1.f / float(params.env_size);
    node.distant = true;
    node.center = equal_area_square_to_sphere((float2(lo) + 0.5f * size) * inv_side);
    float cos_radius = 1.f;
    for (uint corner = 0; corner < 9; corner++) {
        const float2 offset = float2(float(corner % 3), float(corner / 3)) * 0.5f;
        const float3 edge =
            equal_area_square_to_sphere(min((float2(lo) + offset * size) * inv_side, float2(1.f)));
        cos_radius = min(cos_radius, dot(node.center, edge));
    }
    node.radius = cos_radius > 0.f ? sqrt(max(1.f - cos_radius * cos_radius, 0.f)) : 1.f;
    node.spread = min(2.f * (1.f - cos_radius), 1.f);
    return node;
}

// The inverse squared distance and its bounds over the node and the receivers (Walter et al. 2005).
float3
node_geometry(const TreeNode node, const float3 pos, const float reach, const float nearest) {
    if (!(node.flux > 0.f))
        return float3(0.f);
    if (node.distant)
        return float3(1.f, 1.f + node.spread, max(1.f - node.spread, 0.f));
    const float dist = length(pos - node.center);
    const float closest = max(dist - node.radius - reach, nearest);
    const float farthest = dist + node.radius + reach;
    return float3(1.f / max(dist * dist, node.radius * node.radius), 1.f / (closest * closest),
                  1.f / (farthest * farthest));
}

// Mean, mean square and sample count of delivered over nominal radiance, times the node's
// estimated over the sample's inverse squared distance.
typealias Contribution = float4;

Contribution prior_contribution(const float3 geometry, const bool distant) {
    const float spread = geometry.x > 0.f ? 0.5f * (geometry.y - geometry.z) / geometry.x : 0.f;
    const float mean = distant ? LIGHT_GRID_PRIOR_ENV_CONTRIBUTION : LIGHT_GRID_PRIOR_CONTRIBUTION;
    return Contribution(mean, mean * mean * (1.f + spread * spread), LIGHT_GRID_PRIOR_SAMPLES, 0.f);
}

Contribution rescale_contribution(const Contribution c, const float ratio) {
    return Contribution(c.x * ratio, c.y * ratio * ratio, c.z, 0.f);
}

Contribution halve_contribution(const Contribution parent,
                                const float3 parent_geometry,
                                const float3 geometry,
                                const bool distant) {
    const Contribution prior = prior_contribution(geometry, distant);
    if (parent.z <= LIGHT_GRID_PRIOR_SAMPLES || !(geometry.x > 0.f))
        return prior;
    const float mean = parent.x * parent_geometry.x / geometry.x;
    return Contribution(mean, mean * mean * prior.y / (prior.x * prior.x),
                        min(parent.z, 4.f * LIGHT_GRID_PRIOR_SAMPLES), 0.f);
}

struct CutSlot {
    TreeNode node;
    float3 geometry;
    float reach;
    float importance;
    float excess;
};

float nearest_distance(const LightPreprocessBinding params, const CellGeometry cg) {
    return max(0.5f / params.tree_info[0].inv_cell_size, 0.25f * cg.radius);
}

CutSlot cut_slot(const LightPreprocessBinding params,
                 const CellGeometry cg,
                 const uint first,
                 const uint end,
                 const Contribution contribution) {
    CutSlot s;
    s.node = tree_node(params, first, end);
    s.geometry = node_geometry(s.node, cg.center, cg.radius, nearest_distance(params, cg));
    s.reach = s.node.flux * s.geometry.x;
    s.importance = s.reach * max(contribution.x, 0.f);
    s.excess = s.reach * max(contribution.y - contribution.x * contribution.x, 0.f) /
               max(contribution.x, LIGHT_GRID_MIN_CONTRIBUTION);
    return s;
}

void store_cut(const LightPreprocessBinding params,
               const CellGeometry cg,
               const uint cell,
               const uint slot,
               const uint key,
               const uint first,
               const Contribution contribution) {
    const uint lights = light_count(params);
    const bool listed = key != LIGHT_TREE_NO_KEY;
    const uint next_first = WaveShuffle(first, min(slot + 1u, LIGHT_GRID_SLOTS - 1u));
    const uint end = slot < LIGHT_GRID_SLOTS - 1u ? next_first : lights;
    const CutSlot s = cut_slot(params, cg, first, listed ? end : first, contribution);
    const bool emits = listed && s.node.flux > 0.f;
    const float total = WaveActiveSum(emits ? s.importance : 0.f);
    const float even = total > 0.f ? params.grid_even_share : 1.f;
    const uint emitting = WaveActiveCountBits(emits);
    const float p =
        emits ? (1.f - even) * (total > 0.f ? s.importance / total : 0.f) + even / float(emitting)
              : 0.f;

    const uint entry = cell * LIGHT_GRID_SLOTS + slot;
    params.grid_keys[entry] = key;
    params.grid_contribution[entry] = contribution;
    params.grid_starts[entry] = listed ? first : lights;
    params.grid_estimate[entry] = emits && !s.node.distant ? 1.f / s.geometry.x : 1.f;
    for (uint i = 0; i < LIGHT_GRID_FEEDBACK_STRIDE; i++)
        params.grid_feedback[LIGHT_GRID_FEEDBACK_STRIDE * entry + i] = 0;
    if (params.slot_weighing == LightSlotWeighing::LightSlotWeighingShadingPoint) {
        LightBound bound;
        bound.centroid = s.node.center;
        bound.flux = emits ? s.node.flux * max(contribution.x, LIGHT_GRID_MIN_CONTRIBUTION) : 0.f;
        bound.radius = s.node.radius;
        bound.flags = s.node.distant ? uint(EmissiveTriangleFlags::EmissiveDistant) : 0u;
        params.grid_slot_bounds[entry] = bound;
    } else {
        params.grid_probability[entry] = WavePrefixSum(p) + p;
    }
}

groupshared uint s_compact_key[LIGHT_GRID_SLOTS];
groupshared uint s_compact_first[LIGHT_GRID_SLOTS];
groupshared Contribution s_compact_contribution[LIGHT_GRID_SLOTS];

/* Each cell holds a cut through the light tree: a partition of all lights into nodes, drawn by
 * flux over squared distance times their learned contribution, and by flux within a node. A cell
 * splits the node whose lights' contributions vary most and, once full, merges the pair that varies
 * least. Nodes are named by their first Morton code. (Walter et al. 2005; Vevoda et al. 2018.)
 */
[shader("compute")][numthreads(LIGHT_GRID_SLOTS, 1, 1)] void
grid(uint3 group_id : SV_GroupID,
     uint slot : SV_GroupIndex,
     ParameterBlock<LightPreprocessBinding> params) {
    const uint dim = params.grid_info[0].dimension;
    const uint cells_per_cascade = dim * dim * dim;
    const uint cell = group_id.x + group_id.y * LIGHT_GRID_DISPATCH_ROW;
    if (cell >= cells_per_cascade * params.grid_info[0].cascade_count)
        return;

    const CellGeometry cg = cell_geometry(params.grid_info, cell, params.grid_jitter);
    const uint lights = light_count(params);
    if (WaveGetLaneCount() < LIGHT_GRID_SLOTS) {
        store_cut(params, cg, cell, slot, LIGHT_TREE_NO_KEY, lights, Contribution(0.f));
        return;
    }
    const LightGridInfo info = params.grid_info[cg.cascade];
    const uint last_slot = LIGHT_GRID_SLOTS - 1u;
    const uint next_slot = min(slot + 1u, last_slot);

    // carry the cut, or take the coarser cell's
    const bool lattice_kept =
        !params.grid_reset && same_lattice(params.tree_info[0], params.tree_info_prev[0]);
    uint prev_cascade = 0;
    uint3 prev_coord = uint3(0);
    const bool carried = lattice_kept && reproject_cell(params.grid_info_prev, info, cg.coord,
                                                        prev_cascade, prev_coord);
    uint prev_cell =
        prev_cascade * cells_per_cascade + prev_coord.x + dim * (prev_coord.y + dim * prev_coord.z);
    const bool inherited =
        !carried && lattice_kept && coarser_cell(params.grid_info_prev, info, cg.center, prev_cell);

    uint key = LIGHT_TREE_NO_KEY;
    Contribution contribution = Contribution(0.f);
    float samples = 0.f;
    if (carried || inherited) {
        const uint prev_entry = prev_cell * LIGHT_GRID_SLOTS + slot;
        key = params.grid_keys_prev[prev_entry];
        contribution = params.grid_contribution_prev[prev_entry];
        if (carried) {
            const uint feedback_at = LIGHT_GRID_FEEDBACK_STRIDE * prev_entry;
            const uint3 feedback = uint3(params.grid_feedback_prev[feedback_at],
                                         params.grid_feedback_prev[feedback_at + 1],
                                         params.grid_feedback_prev[feedback_at + 2]);
            samples = float(min(feedback.x, LIGHT_GRID_FEEDBACK_SAMPLES));
            if (samples > 0.f) {
                const float2 sums =
                    float2(feedback.yz) /
                    float2(LIGHT_GRID_CONTRIBUTION_SCALE, LIGHT_GRID_CONTRIBUTION_SQUARE_SCALE);
                contribution.xy =
                    (sums + contribution.z * contribution.xy) / (samples + contribution.z);
                contribution.z = min(contribution.z + samples, LIGHT_GRID_CONTRIBUTION_MEMORY);
            }
        }
    }

    // a carried cut must still partition all lights
    bool fresh = !(carried || inherited);
    {
        const uint next_key = WaveShuffle(key, next_slot);
        const bool listed = key != LIGHT_TREE_NO_KEY;
        const bool env = listed && key >= LIGHT_TREE_ENV_KEY;
        const uint prev_key = WaveShuffle(key, slot > 0 ? slot - 1u : 0u);
        bool valid = slot != 0 || key == 0;
        if (slot < last_slot)
            valid &= listed ? next_key == LIGHT_TREE_NO_KEY || next_key > key
                            : next_key == LIGHT_TREE_NO_KEY;
        if (env) {
            valid &= key - LIGHT_TREE_ENV_KEY < env_texels(params);
            if (slot == 0 || prev_key < LIGHT_TREE_ENV_KEY)
                valid &= key == LIGHT_TREE_ENV_KEY;
        }
        if (!WaveActiveAllTrue(valid) || WaveActiveAnyTrue(env) != (env_texels(params) > 0)) {
            key = slot == 0                              ? 0u
                  : slot == 1 && env_texels(params) > 0u ? LIGHT_TREE_ENV_KEY
                                                         : LIGHT_TREE_NO_KEY;
            fresh = true;
        }
    }

    uint first = node_start(params, key);
    float flux;
    {
        const uint next_first = WaveShuffle(first, next_slot);
        const uint end = slot < last_slot ? next_first : lights;
        const TreeNode node = tree_node(params, first, key != LIGHT_TREE_NO_KEY ? end : first);
        const float nearest = nearest_distance(params, cg);
        const float3 geometry = node_geometry(node, cg.center, cg.radius, nearest);
        flux = node.flux;
        if (fresh) {
            contribution = prior_contribution(geometry, node.distant);
        } else if (inherited && geometry.x > 0.f) {
            const LightGridInfo coarser = params.grid_info_prev[prev_cell / cells_per_cascade];
            const float3 coarser_center =
                coarser.origin +
                (floor((cg.center - coarser.origin) * coarser.inv_cell_size) + 0.5f) /
                    coarser.inv_cell_size;
            const float3 there = node_geometry(node, coarser_center, 0.f, nearest);
            contribution = rescale_contribution(contribution, there.x / geometry.x);
            contribution.z = min(contribution.z, LIGHT_GRID_PRIOR_SAMPLES);
        }
    }

    // merge non-emitting and dead nodes into their predecessor
    {
        const bool listed = key != LIGHT_TREE_NO_KEY;
        const bool dead = listed && !(contribution.x > 0.f) && !(contribution.y > 0.f) &&
                          contribution.z >= LIGHT_GRID_DEAD_SAMPLES;
        const bool previous_dead = WaveShuffle(dead, slot > 0 ? slot - 1u : 0u);
        const uint previous_key = WaveShuffle(key, slot > 0 ? slot - 1u : 0u);
        const bool absorbed = slot > 0 && listed && (!(flux > 0.f) || (dead && previous_dead)) &&
                              (key >= LIGHT_TREE_ENV_KEY) == (previous_key >= LIGHT_TREE_ENV_KEY);
        const bool keep = listed && !absorbed;
        if (WaveActiveAnyTrue(absorbed)) {
            const uint at = WavePrefixCountBits(keep);
            if (keep) {
                s_compact_key[at] = key;
                s_compact_first[at] = first;
                s_compact_contribution[at] = contribution;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint kept = WaveActiveCountBits(keep);
            key = slot < kept ? s_compact_key[slot] : LIGHT_TREE_NO_KEY;
            first = slot < kept ? s_compact_first[slot] : lights;
            contribution = slot < kept ? s_compact_contribution[slot] : Contribution(0.f);
        }
    }

    // refine
    const uint steps = fresh                              ? LIGHT_GRID_FRESH_SPLITS
                       : !carried                         ? LIGHT_GRID_SLOTS
                       : WaveActiveAnyTrue(samples > 0.f) ? params.grid_refinements
                                                          : 0u;
    for (uint step = 0; step < steps; step++) {
        const bool listed = key != LIGHT_TREE_NO_KEY;
        const uint next_first = WaveShuffle(first, next_slot);
        const uint end = slot < last_slot ? next_first : lights;
        const CutSlot s = cut_slot(params, cg, first, listed ? end : first, contribution);

        const float candidate = s.node.splittable ? s.excess : 0.f;
        const float best = WaveActiveMax(candidate);
        if (!(best > 0.f))
            break;
        const uint split_slot = firstbitlow(WaveActiveBallot(candidate == best).x);

        const uint next_key = WaveShuffle(key, next_slot);
        const uint next_end = WaveShuffle(end, next_slot);
        const float next_excess = WaveShuffle(s.excess, next_slot);
        const float next_flux = WaveShuffle(s.node.flux, next_slot);
        const float next_estimate = WaveShuffle(s.geometry.x, next_slot);
        const Contribution next_contribution = WaveShuffle(contribution, next_slot);
        const bool pair = slot < last_slot && listed && next_key != LIGHT_TREE_NO_KEY &&
                          (key >= LIGHT_TREE_ENV_KEY) == (next_key >= LIGHT_TREE_ENV_KEY);
        float merge_loss = 0.f;
        Contribution merged_contribution = contribution;
        if (pair) {
            const TreeNode merged_node = tree_node(params, first, next_end);
            const float merged_estimate =
                node_geometry(merged_node, cg.center, cg.radius, nearest_distance(params, cg)).x;
            const float merged_flux = s.node.flux + next_flux;
            const float w = merged_flux > 0.f ? s.node.flux / merged_flux : 0.5f;
            const Contribution a =
                merged_estimate > 0.f
                    ? rescale_contribution(contribution, s.geometry.x / merged_estimate)
                    : contribution;
            const Contribution b =
                merged_estimate > 0.f
                    ? rescale_contribution(next_contribution, next_estimate / merged_estimate)
                    : next_contribution;
            merged_contribution = Contribution(w * a.xy + (1.f - w) * b.xy, min(a.z, b.z), 0.f);
            const CutSlot merged = cut_slot(params, cg, first, next_end, merged_contribution);
            merge_loss = max(merged.excess - s.excess - next_excess, 0.f);
        }

        uint split_first = 0;
        Contribution low_contribution = contribution;
        Contribution high_contribution = contribution;
        if (slot == split_slot) {
            split_first = s.node.split_key >= LIGHT_TREE_ENV_KEY
                              ? params.triangle_count + (s.node.split_key - LIGHT_TREE_ENV_KEY)
                              : lower_bound(params, s.node.split_key, first, end);
            const CutSlot low = cut_slot(params, cg, first, split_first, contribution);
            const CutSlot high = cut_slot(params, cg, split_first, end, contribution);
            low_contribution =
                halve_contribution(contribution, s.geometry, low.geometry, s.node.distant);
            high_contribution =
                halve_contribution(contribution, s.geometry, high.geometry, s.node.distant);
        }
        const float gain = 0.5f * best;
        split_first = WaveReadLaneAt(split_first, split_slot);
        const uint split_key = WaveReadLaneAt(s.node.split_key, split_slot);
        low_contribution = WaveReadLaneAt(low_contribution, split_slot);
        high_contribution = WaveReadLaneAt(high_contribution, split_slot);

        uint merge_slot = LIGHT_GRID_SLOTS;
        if (!WaveActiveAnyTrue(!listed)) {
            const bool mergeable = pair && slot != split_slot && slot + 1u != split_slot;
            const float cheapest = WaveActiveMin(mergeable ? merge_loss : MERIAN_FLT_MAX);
            if (!WaveActiveAnyTrue(mergeable) || !(gain > LIGHT_GRID_REFINE_MARGIN * cheapest))
                break;
            merge_slot = firstbitlow(WaveActiveBallot(mergeable && merge_loss == cheapest).x);
        }

        uint target = split_slot;
        if (merge_slot < LIGHT_GRID_SLOTS) {
            const Contribution merged = WaveReadLaneAt(merged_contribution, merge_slot);
            const uint from = slot > merge_slot ? next_slot : slot;
            key = WaveShuffle(key, from);
            first = WaveShuffle(first, from);
            contribution = WaveShuffle(contribution, from);
            if (slot == last_slot) {
                key = LIGHT_TREE_NO_KEY;
                first = lights;
            }
            if (slot == merge_slot)
                contribution = merged;
            target -= target > merge_slot ? 1u : 0u;
        }

        const uint from = slot > target + 1u ? slot - 1u : slot;
        key = WaveShuffle(key, from);
        first = WaveShuffle(first, from);
        contribution = WaveShuffle(contribution, from);
        if (slot == target)
            contribution = low_contribution;
        if (slot == target + 1u) {
            key = split_key;
            first = split_first;
            contribution = high_contribution;
        }
    }

    store_cut(params, cg, cell, slot, key, first, contribution);
}
