#include "merian-shaders/utils/math-constants.slangh"

import merian_shaders.scene.env_importance;
import merian_shaders.scene.light_data;
import merian_shaders.utils.envmap;
import merian_shaders.utils.morton;

using merian;

// a block whose cell a lookup found within these frames gives way only to finer cells of its class
// and to the classes before it
static const uint LIGHT_GRID_RECENT_FRAMES = 8;
static const uint LIGHT_GRID_IDLE_RANKS = 64;
static const uint LIGHT_GRID_RECENT_RANKS =
    uint(LightGridClass::LightGridClassCount) * LIGHT_GRID_MAX_CASCADES;
static const uint LIGHT_GRID_EVICTION_RANKS = LIGHT_GRID_RECENT_RANKS + LIGHT_GRID_IDLE_RANKS;

// Per-frame preprocess for next event estimation: cuts through the light tree for the cells of a
// world-space grid that rays look up, and one over the whole scene.

struct LightPreprocessBinding {
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    StructuredBuffer<uint> tree_keys;
    StructuredBuffer<float2> tree_cdf;
    // size 0 where the cells do not list the environment
    EnvImportance env;
    // per node of the environment's texels down to ENV_CONE_LOG2, heap-indexed: the cone around
    // them as (center, cosine of the radius)
    RWStructuredBuffer<float4> env_cones;
    RWStructuredBuffer<LightTreeInfo, ScalarDataLayout> tree_info;
    StructuredBuffer<LightTreeInfo, ScalarDataLayout> tree_info_prev;
    RWStructuredBuffer<uint> grid_keys;
    StructuredBuffer<uint> grid_keys_prev;
    RWStructuredBuffer<float3, ScalarDataLayout> grid_contribution;
    StructuredBuffer<float3, ScalarDataLayout> grid_contribution_prev;
    RWStructuredBuffer<float> grid_estimate;
    RWStructuredBuffer<uint> grid_feedback;
    StructuredBuffer<uint> grid_feedback_prev;
    RWStructuredBuffer<uint> grid_starts;
    RWStructuredBuffer<float2> grid_env_prefix;
    RWStructuredBuffer<float> grid_probability;
    RWStructuredBuffer<LightBound, ScalarDataLayout> grid_slot_bounds;
    RWStructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info;
    StructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info_prev;
    RWStructuredBuffer<uint> grid_index;
    StructuredBuffer<uint> grid_index_prev;
    RWStructuredBuffer<LightGridBlock, ScalarDataLayout> grid_blocks;
    RWStructuredBuffer<uint> grid_free;
    // per class and allocate group its missed cells, then their first free block; then the free
    // block count, then per class and cascade the missed cells
    RWStructuredBuffer<uint> grid_offsets;
    // zeroed, see LIGHT_GRID_SETUP_STATE
    RWStructuredBuffer<uint> setup_state;

    // fraction of the light bounds the grid spans around the camera
    float grid_coverage;
    float3 camera_position;
    uint triangle_count;
    uint grid_dimension;
    uint grid_cascades;
    float grid_cell_size;
    float grid_jitter;
    float grid_even_share;
    uint grid_refinements;
    uint cell_count;
    uint block_count;
    uint allocate_groups;
    float triangle_prior;
    float env_prior;
    uint setup_groups;
    LightSlotWeighing slot_weighing;
    // drop everything carried over, for a frame the previous grid cannot describe
    bool grid_reset;
};

static const uint SETUP_DONE = LIGHT_GRID_SETUP_STATE - 1;
groupshared float3 s_setup_min[LIGHT_GRID_SETUP_GROUP];
groupshared float3 s_setup_max[LIGHT_GRID_SETUP_GROUP];
groupshared float s_setup_distance[LIGHT_GRID_SETUP_GROUP];
groupshared bool s_setup_last;

// Floats as unsigned integers of the same order; zero is below every float.
uint ordered(const float f) {
    const uint u = asuint(f);
    return (u & 0x80000000u) != 0 ? ~u : u | 0x80000000u;
}

float from_ordered(const uint u) {
    return asfloat((u & 0x80000000u) != 0 ? u & 0x7FFFFFFFu : ~u);
}

float wide_sum(const StructuredBuffer<uint> words, const uint at) {
    return float(words[at + 1]) * 4294967296.f + float(words[at]);
}

void merge_maximum(const LightPreprocessBinding params, const uint at, const float value) {
    uint previous;
    InterlockedMax(params.setup_state[at], ordered(value), previous);
}

float read_maximum(const LightPreprocessBinding params, const uint at) {
    uint value;
    InterlockedOr(params.setup_state[at], 0u, value);
    return from_ordered(value);
}

bool same_lattice(const LightTreeInfo a, const LightTreeInfo b) {
    return all(asuint(a.origin) == asuint(b.origin)) &&
           asuint(a.inv_cell_size) == asuint(b.inv_cell_size);
}

// Fits the light tree's lattice to the lights and places the cascades around the camera. The
// outermost one has to reach the lights the camera can see, so its half extent comes from the
// distance from the camera to the farthest light, with the spread of the lights as a floor for a
// camera standing among them. Every finer cascade halves the cell size, so a cell covers a roughly
// constant angle from the camera at any scale.
// The groups merge their bounds, and the last one to finish places everything.
[shader("compute")][numthreads(LIGHT_GRID_SETUP_GROUP, 1, 1)] void
grid_setup(uint3 group_id : SV_GroupID,
           uint local_id : SV_GroupIndex,
           ParameterBlock<LightPreprocessBinding> params) {
    float3 bmin = float3(1e30f);
    float3 bmax = float3(-1e30f);
    float max_distance = 0.f;
    const uint base = group_id.x * LIGHT_GRID_SETUP_GROUP * LIGHT_GRID_SETUP_ITEMS;
    for (uint item = 0; item < LIGHT_GRID_SETUP_ITEMS; item++) {
        const uint i = base + item * LIGHT_GRID_SETUP_GROUP + local_id;
        if (i >= params.triangle_count)
            break;
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

    if (local_id == 0) {
        const uint wave_count = LIGHT_GRID_SETUP_GROUP / WaveGetLaneCount();
        for (uint w = 0; w < wave_count; w++) {
            bmin = min(bmin, s_setup_min[w]);
            bmax = max(bmax, s_setup_max[w]);
            max_distance = max(max_distance, s_setup_distance[w]);
        }
        for (uint axis = 0; axis < 3; axis++) {
            merge_maximum(params, axis, bmax[axis]);
            merge_maximum(params, 3 + axis, -bmin[axis]);
        }
        merge_maximum(params, 6, max_distance);
        AllMemoryBarrier();
        uint done;
        InterlockedAdd(params.setup_state[SETUP_DONE], 1u, done);
        s_setup_last = done + 1u == params.setup_groups;
    }
    GroupMemoryBarrierWithGroupSync();
    if (!s_setup_last)
        return;
    AllMemoryBarrier();
    for (uint axis = 0; axis < 3; axis++) {
        bmax[axis] = read_maximum(params, axis);
        bmin[axis] = -read_maximum(params, 3 + axis);
    }
    max_distance = read_maximum(params, 6);

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
    float scale;
    uint cascade;
};

CellGeometry cell_geometry(const RWStructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info,
                           const uint cascade,
                           const uint3 coord,
                           const float jitter) {
    CellGeometry cg;
    cg.cascade = cascade;
    const LightGridInfo info = grid_info[cascade];
    const float cell_size = 1.f / info.inv_cell_size;
    cg.center = info.origin + (float3(coord) + 0.5f) * cell_size;
    // the sphere the cell stands for: its circumradius, grown by the jitter the lookup adds
    cg.radius = 0.5f * sqrt(3.f) * cell_size * (jitter + 1.f);
    cg.scale = light_cell_scale(info.inv_cell_size);
    return cg;
}

CellGeometry
global_cell_geometry(const RWStructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info) {
    CellGeometry cg;
    cg.cascade = grid_info[0].cascade_count - 1u;
    const LightGridInfo outermost = grid_info[cg.cascade];
    const float extent = float(outermost.dimension) / outermost.inv_cell_size;
    cg.center = outermost.origin + 0.5f * extent;
    cg.radius = 0.5f * sqrt(3.f) * extent;
    cg.scale = light_cell_scale(outermost.inv_cell_size);
    return cg;
}

int3 cascade_offset(const LightGridInfo info) {
    return int3(floor(info.origin * info.inv_cell_size + 0.25f));
}

bool block_cell(const LightPreprocessBinding params,
                const LightGridBlock b,
                out uint cascade,
                out uint3 local) {
    cascade = 0;
    local = uint3(0);
    const LightGridInfo finest = params.grid_info[0];
    if (!(finest.inv_cell_size > 0.f) || !(b.inv_cell_size > 0.f))
        return false;
    const int c = int(round(log2(finest.inv_cell_size / b.inv_cell_size)));
    if (c < 0 || c >= int(finest.cascade_count))
        return false;
    const LightGridInfo info = params.grid_info[c];
    if (abs(info.inv_cell_size - b.inv_cell_size) > 1e-6f * b.inv_cell_size)
        return false;
    const int3 l = b.coord - cascade_offset(info);
    if (any(l < 0) || any(l >= int(info.dimension)))
        return false;
    cascade = uint(c);
    local = uint3(l);
    return true;
}

CellGeometry block_geometry(const LightPreprocessBinding params, const LightGridBlock b) {
    uint cascade;
    uint3 coord;
    block_cell(params, b, cascade, coord);
    return cell_geometry(params.grid_info, cascade, coord, params.grid_jitter);
}

bool coarser_cell(const StructuredBuffer<LightGridInfo, ScalarDataLayout> grid_info_prev,
                  const LightGridInfo info,
                  const float3 pos,
                  const uint orientation,
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
    if (any(c < 0) || any(c >= int(info.dimension)))
        return false;
    prev_cell = light_cell_index(uint(cascade), uint3(c), orientation, info.dimension);
    return true;
}

groupshared uint s_prefix_wave_totals[LIGHT_GRID_ALLOCATE_GROUP];

uint group_prefix_sum(const uint local_id, const uint value, out uint total) {
    const uint lanes = WaveGetLaneCount();
    const uint wave = local_id / lanes;
    const uint in_wave = WavePrefixSum(value);
    const uint wave_total = WaveActiveSum(value);
    if (WaveIsFirstLane())
        s_prefix_wave_totals[wave] = wave_total;
    GroupMemoryBarrierWithGroupSync();
    uint before = 0;
    total = 0;
    for (uint w = 0; w < LIGHT_GRID_ALLOCATE_GROUP / lanes; w++) {
        const uint count = s_prefix_wave_totals[w];
        before += w < wave ? count : 0u;
        total += count;
    }
    GroupMemoryBarrierWithGroupSync();
    return before + in_wave;
}

LightGridBlock prev_cell_block(const LightPreprocessBinding params, const uint prev_cell) {
    const uint dim = params.grid_dimension;
    const LightGridInfo prev = params.grid_info_prev[light_cell_cascade(prev_cell, dim)];
    LightGridBlock b;
    b.coord = int3(light_cell_coord(prev_cell, dim)) + cascade_offset(prev);
    b.inv_cell_size = prev.inv_cell_size;
    b.orientation = light_cell_orientation(prev_cell);
    b.idle = 0;
    b.kept = 0;
    b.unseen = 0;
    return b;
}

bool moved_cell(const LightPreprocessBinding params, const LightGridBlock b, out uint cell) {
    cell = 0;
    uint cascade;
    uint3 coord;
    if (!block_cell(params, b, cascade, coord))
        return false;
    cell = light_cell_index(cascade, coord, b.orientation, params.grid_dimension);
    return true;
}

uint prev_index_entry(const LightPreprocessBinding params, const uint prev_cell) {
    return prev_cell < params.cell_count && !params.grid_reset ? params.grid_index_prev[prev_cell]
                                                               : 0u;
}

LightGridClass light_grid_class(const bool seen) {
    return seen ? LightGridClass::LightGridClassSeen : LightGridClass::LightGridClassOther;
}

uint light_grid_bucket(const bool seen, const uint cascade) {
    return uint(light_grid_class(seen)) * LIGHT_GRID_MAX_CASCADES + cascade;
}

uint missed_group_slot(const LightPreprocessBinding params, const bool seen, const uint group) {
    return uint(light_grid_class(seen)) * params.allocate_groups + group;
}

uint free_count_slot(const LightPreprocessBinding params) {
    return uint(LightGridClass::LightGridClassCount) * params.allocate_groups;
}

uint missed_bucket_slot(const LightPreprocessBinding params, const uint bucket) {
    return free_count_slot(params) + 1u + bucket;
}

uint missed_counts(const bool missed, const bool seen) {
    return missed ? (seen ? 1u : 0x10000u) : 0u;
}

// Carries each held block to its cell in this frame's index and counts per class the visited cells
// without one.
[shader("compute")][numthreads(LIGHT_GRID_ALLOCATE_GROUP, 1, 1)] void
grid_keep(uint3 group_id : SV_GroupID,
          uint local_id : SV_GroupIndex,
          ParameterBlock<LightPreprocessBinding> params) {
    const uint prev_cell = group_id.x * LIGHT_GRID_ALLOCATE_GROUP + local_id;
    const uint entry = prev_index_entry(params, prev_cell);
    const uint block = light_cell_block(entry);
    const bool visited = (entry & LIGHT_GRID_VISITED) != 0;
    const bool seen = (entry & LIGHT_GRID_SEEN) != 0;
    uint cell = 0;
    const bool moved = entry != 0 && moved_cell(params, prev_cell_block(params, prev_cell), cell);
    if (block != LIGHT_GRID_FALLBACK_BLOCK) {
        if (moved) {
            params.grid_index[cell] = block;
            params.grid_blocks[block].idle =
                visited ? 0u
                        : min(params.grid_blocks[block].idle + 1u,
                              LIGHT_GRID_RECENT_FRAMES + LIGHT_GRID_IDLE_RANKS - 1u);
            params.grid_blocks[block].kept = 1;
            params.grid_blocks[block].unseen =
                seen ? 0u : min(params.grid_blocks[block].unseen + 1u, LIGHT_GRID_RECENT_FRAMES);
        } else {
            params.grid_blocks[block].idle = LIGHT_GRID_IDLE_FREE;
        }
    }
    const bool missed = visited && block == LIGHT_GRID_FALLBACK_BLOCK && moved;
    if (missed) {
        const uint cascade = light_cell_cascade(cell, params.grid_dimension);
        const uint bucket = light_grid_bucket(seen, cascade);
        InterlockedAdd(params.grid_offsets[missed_bucket_slot(params, bucket)], 1u);
    }
    uint totals;
    group_prefix_sum(local_id, missed_counts(missed, seen), totals);
    if (local_id == 0) {
        params.grid_offsets[missed_group_slot(params, true, group_id.x)] = totals & 0xFFFFu;
        params.grid_offsets[missed_group_slot(params, false, group_id.x)] = totals >> 16;
    }
}

bool held_block(const LightPreprocessBinding params, const uint block) {
    return block < params.block_count && block != LIGHT_GRID_FALLBACK_BLOCK && !params.grid_reset &&
           params.grid_blocks[block].idle != LIGHT_GRID_IDLE_FREE;
}

// Higher goes first: the longest unvisited, then the recently visited by class, the coarsest first.
uint eviction_rank(const LightPreprocessBinding params, const uint block, out uint cell) {
    const LightGridBlock b = params.grid_blocks[block];
    uint cascade;
    uint3 coord;
    block_cell(params, b, cascade, coord);
    cell = light_cell_index(cascade, coord, b.orientation, params.grid_dimension);
    if (b.idle >= LIGHT_GRID_RECENT_FRAMES)
        return LIGHT_GRID_RECENT_RANKS + b.idle - LIGHT_GRID_RECENT_FRAMES;
    return light_grid_bucket(b.unseen < LIGHT_GRID_RECENT_FRAMES, cascade);
}

groupshared uint s_rank_blocks[LIGHT_GRID_EVICTION_RANKS];

[shader("compute")][numthreads(LIGHT_GRID_ALLOCATE_GROUP, 1, 1)] void
grid_allocate(uint local_id : SV_GroupIndex, ParameterBlock<LightPreprocessBinding> params) {
    // a reset frees every block, even without missed cells
    uint missed_cells = 0;
    for (uint bucket = 0; bucket < LIGHT_GRID_RECENT_RANKS; bucket++)
        missed_cells += params.grid_offsets[missed_bucket_slot(params, bucket)];
    if (missed_cells == 0 && !params.grid_reset)
        return;
    for (uint rank = local_id; rank < LIGHT_GRID_EVICTION_RANKS; rank += LIGHT_GRID_ALLOCATE_GROUP)
        s_rank_blocks[rank] = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint block = local_id; block < params.block_count; block += LIGHT_GRID_ALLOCATE_GROUP) {
        uint cell;
        if (held_block(params, block))
            InterlockedAdd(s_rank_blocks[eviction_rank(params, block, cell)], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    // the missed cells take free blocks, then held ones ranked after their class and cascade, class
    // by class and the nearest first
    uint held = 0;
    uint idle_held = 0;
    for (uint rank = 0; rank < LIGHT_GRID_EVICTION_RANKS; rank++) {
        held += s_rank_blocks[rank];
        idle_held += rank >= LIGHT_GRID_RECENT_RANKS ? s_rank_blocks[rank] : 0u;
    }
    const uint unheld = params.block_count - 1u - held;
    uint outranked = held - idle_held;
    uint taken = 0;
    for (uint bucket = 0; bucket < LIGHT_GRID_RECENT_RANKS; bucket++) {
        outranked -= s_rank_blocks[bucket];
        const uint takeable = unheld + idle_held + outranked;
        taken += min(params.grid_offsets[missed_bucket_slot(params, bucket)],
                     takeable > taken ? takeable - taken : 0u);
    }
    const uint evicted = taken > unheld ? taken - unheld : 0u;
    uint threshold = LIGHT_GRID_EVICTION_RANKS;
    uint above = 0;
    while (threshold > 0 && above + s_rank_blocks[threshold - 1u] <= evicted)
        above += s_rank_blocks[--threshold];
    const uint tie_rank = threshold > 0 ? threshold - 1u : 0u;
    const uint ties_evicted = evicted - above;

    uint free_count = 0;
    uint ties_seen = 0;
    for (uint base = 0; base < params.block_count;
         base += LIGHT_GRID_ALLOCATE_GROUP * LIGHT_GRID_ALLOCATE_ITEMS) {
        const uint first = base + local_id * LIGHT_GRID_ALLOCATE_ITEMS;
        uint ranks[LIGHT_GRID_ALLOCATE_ITEMS];
        uint cells[LIGHT_GRID_ALLOCATE_ITEMS];
        uint ties = 0;
        for (uint i = 0; i < LIGHT_GRID_ALLOCATE_ITEMS; i++) {
            const uint block = first + i;
            ranks[i] = held_block(params, block) ? eviction_rank(params, block, cells[i])
                                                 : LIGHT_GRID_EVICTION_RANKS;
            ties += uint(ranks[i] == tie_rank);
        }
        uint ties_total;
        uint tie = ties_seen + group_prefix_sum(local_id, ties, ties_total);
        ties_seen += ties_total;

        uint frees = 0;
        uint free_items = 0;
        for (uint i = 0; i < LIGHT_GRID_ALLOCATE_ITEMS; i++) {
            const uint block = first + i;
            const bool usable = block < params.block_count && block != LIGHT_GRID_FALLBACK_BLOCK;
            bool free = usable && ranks[i] >= threshold;
            if (usable && ranks[i] == tie_rank) {
                free = tie < ties_evicted;
                tie++;
            }
            frees |= uint(free) << i;
            free_items += uint(free);
        }
        uint total;
        uint at = free_count + group_prefix_sum(local_id, free_items, total);
        free_count += total;
        for (uint i = 0; i < LIGHT_GRID_ALLOCATE_ITEMS; i++) {
            if ((frees & (1u << i)) == 0)
                continue;
            const uint block = first + i;
            params.grid_free[at++] = block;
            if (ranks[i] < LIGHT_GRID_EVICTION_RANKS)
                params.grid_index[cells[i]] = LIGHT_GRID_FALLBACK_BLOCK;
            params.grid_blocks[block].idle = LIGHT_GRID_IDLE_FREE;
        }
    }

    uint missed_count = 0;
    for (uint base = 0; base < free_count_slot(params); base += LIGHT_GRID_ALLOCATE_GROUP) {
        const uint slot = base + local_id;
        const bool listed = slot < free_count_slot(params);
        uint total;
        const uint offset =
            missed_count +
            group_prefix_sum(local_id, listed ? params.grid_offsets[slot] : 0u, total);
        if (listed)
            params.grid_offsets[slot] = offset;
        missed_count += total;
    }
    if (local_id == 0)
        params.grid_offsets[free_count_slot(params)] = free_count;
}

bool missed_cell(const LightPreprocessBinding params,
                 const uint prev_cell,
                 out LightGridBlock b,
                 out uint cell,
                 out bool seen) {
    cell = 0;
    b = {};
    const uint entry = prev_index_entry(params, prev_cell);
    seen = (entry & LIGHT_GRID_SEEN) != 0;
    if ((entry & LIGHT_GRID_VISITED) == 0 || light_cell_block(entry) != LIGHT_GRID_FALLBACK_BLOCK)
        return false;
    b = prev_cell_block(params, prev_cell);
    b.unseen = seen ? 0u : LIGHT_GRID_RECENT_FRAMES;
    return moved_cell(params, b, cell);
}

[shader("compute")][numthreads(LIGHT_GRID_ALLOCATE_GROUP, 1, 1)] void
grid_assign(uint3 group_id : SV_GroupID,
            uint local_id : SV_GroupIndex,
            ParameterBlock<LightPreprocessBinding> params) {
    LightGridBlock b;
    uint cell;
    bool seen;
    const bool missed =
        missed_cell(params, group_id.x * LIGHT_GRID_ALLOCATE_GROUP + local_id, b, cell, seen);
    uint totals;
    const uint before = group_prefix_sum(local_id, missed_counts(missed, seen), totals);
    const uint at = params.grid_offsets[missed_group_slot(params, seen, group_id.x)] +
                    (seen ? before & 0xFFFFu : before >> 16);
    if (missed && at < params.grid_offsets[free_count_slot(params)]) {
        const uint block = params.grid_free[at];
        params.grid_index[cell] = block;
        params.grid_blocks[block] = b;
    }
}

uint env_texels(const LightPreprocessBinding params) {
    return params.env.texel_count();
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

    bool is_empty() {
        return !distant && !(radius > 0.f);
    }
};

// nodes below the cones' grid take their ancestor's
uint env_cone_index(const uint side, const uint first_texel, const uint free_bits) {
    const uint bits = max(free_bits, env_grid_shift(side, ENV_CONE_LOG2));
    return (1u << (2u * firstbithigh(side) - bits)) + (first_texel >> bits);
}

[shader("compute")][numthreads(64, 1, 1)] void
env_cones(uint3 thread_id : SV_DispatchThreadID, ParameterBlock<LightPreprocessBinding> params) {
    const uint index = thread_id.x + 1u;
    const uint side_bits = 2u * firstbithigh(params.env.size);
    if (index >= 2u << (side_bits - env_grid_shift(params.env.size, ENV_CONE_LOG2)))
        return;
    const uint level_bits = firstbithigh(index);
    const uint free_bits = side_bits - level_bits;
    const uint first_texel = (index - (1u << level_bits)) << free_bits;

    const uint2 axis_free = uint2((free_bits + 1u) / 2u, free_bits / 2u);
    const uint2 lo = (morton_decode2d(first_texel) >> axis_free) << axis_free;
    const float2 size = float2(uint2(1u) << axis_free);
    const float inv_side = 1.f / float(params.env.size);
    const float3 center = equal_area_square_to_sphere((float2(lo) + 0.5f * size) * inv_side);
    float cos_radius = 1.f;
    for (uint corner = 0; corner < 9; corner++) {
        const float2 offset = float2(float(corner % 3), float(corner / 3)) * 0.5f;
        const float3 edge =
            equal_area_square_to_sphere(min((float2(lo) + offset * size) * inv_side, float2(1.f)));
        cos_radius = min(cos_radius, dot(center, edge));
    }
    params.env_cones[index] = float4(center, cos_radius);
}

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
    node.flux =
        max(ff_diff(params.env.prefix(last_texel + 1u), params.env.prefix(first_texel)), 0.f) *
        (MERIAN_PI * MERIAN_FOUR_PI / texels);
    const uint free_bits = shared_prefix_free_bits(first_texel, last_texel);
    node.splittable = free_bits > 0;
    if (node.splittable)
        node.split_key =
            LIGHT_TREE_ENV_KEY + ((last_texel >> (free_bits - 1u)) << (free_bits - 1u));
    const float4 cone = params.env_cones[env_cone_index(params.env.size, first_texel, free_bits)];
    node.distant = true;
    node.center = cone.xyz;
    const float cos_radius = cone.w;
    node.radius = cos_radius > 0.f ? sqrt(max(1.f - cos_radius * cos_radius, 0.f)) : 1.f;
    node.spread = min(2.f * (1.f - cos_radius), 1.f);
    return node;
}

// The inverse squared distance in cell sizes and its bounds over the node and the receivers (Walter
// et al. 2005).
float3 node_geometry(const TreeNode node,
                     const float3 pos,
                     const float reach,
                     const float nearest,
                     const float scale) {
    if (node.is_empty())
        return float3(0.f);
    if (node.distant)
        return scale * float3(1.f, 1.f + node.spread, max(1.f - node.spread, 0.f));
    const float dist = length(pos - node.center);
    const float closest = max(dist - node.radius - reach, nearest);
    const float farthest = dist + node.radius + reach;
    return scale * float3(1.f / max(dist * dist, node.radius * node.radius),
                          1.f / (closest * closest), 1.f / (farthest * farthest));
}

// Mean, mean square and sample count of delivered over nominal radiance, times the sample's
// inverse squared distance in cell sizes.
typealias Contribution = float3;

float prior_mean(const LightPreprocessBinding params, const bool distant) {
    return distant ? params.env_prior : params.triangle_prior;
}

float spread_moment(const float3 geometry) {
    const float spread = geometry.x > 0.f ? 0.5f * (geometry.y - geometry.z) / geometry.x : 0.f;
    return 1.f + spread * spread;
}

Contribution rescale_contribution(const Contribution c, const float ratio) {
    return Contribution(c.xy * float2(ratio, ratio * ratio), c.z);
}

float2 relative_contribution(const Contribution c, const float3 geometry) {
    const float inv_geometry = geometry.x > 0.f ? 1.f / geometry.x : 0.f;
    return float2(c.x * inv_geometry, c.y * inv_geometry * inv_geometry);
}

Contribution prior_contribution(const float3 geometry, const float mean) {
    const float m = mean * geometry.x;
    return Contribution(m, m * m * spread_moment(geometry), LIGHT_GRID_PRIOR_SAMPLES);
}

Contribution
halve_contribution(const Contribution parent, const float3 geometry, const float prior) {
    if (parent.z <= LIGHT_GRID_PRIOR_SAMPLES)
        return prior_contribution(geometry, prior);
    return Contribution(parent.x, parent.x * parent.x * spread_moment(geometry),
                        min(parent.z, 4.f * LIGHT_GRID_PRIOR_SAMPLES));
}

struct CutSlot {
    TreeNode node;
    float3 geometry;
    float importance;
    float excess;
};

float nearest_distance(const LightPreprocessBinding params, const CellGeometry cg) {
    return max(0.5f / params.tree_info[0].inv_cell_size, 0.25f * cg.radius);
}

CutSlot cut_slot(const TreeNode node, const float3 geometry, const Contribution contribution) {
    CutSlot s;
    s.node = node;
    s.geometry = geometry;
    const float2 relative = relative_contribution(contribution, geometry);
    s.importance = s.node.flux * max(contribution.x, 0.f);
    s.excess = s.node.flux * s.geometry.x * max(relative.y - relative.x * relative.x, 0.f) /
               max(relative.x, LIGHT_GRID_MIN_CONTRIBUTION);
    return s;
}

// The slots of a cell, one thread each; every slot enters every call.
interface ISlotGroup {
    float sum(const float value);
    float maximum(const float value);
    float minimum(const float value);
    uint count(const bool value);
    bool any_true(const bool value);
    bool all_true(const bool value);
    float prefix_sum(const float value);
    uint prefix_count(const bool value);
    uint first(const bool value);
    uint read(const uint value, const uint slot);
    float read(const float value, const uint slot);
    float3 read(const float3 value, const uint slot);
    bool read(const bool value, const uint slot);
    uint broadcast(const uint value, const uint slot);
    float3 broadcast(const float3 value, const uint slot);
}

struct WaveSlots : ISlotGroup {
    float sum(const float value) {
        return WaveActiveSum(value);
    }
    float maximum(const float value) {
        return WaveActiveMax(value);
    }
    float minimum(const float value) {
        return WaveActiveMin(value);
    }
    uint count(const bool value) {
        return WaveActiveCountBits(value);
    }
    bool any_true(const bool value) {
        return WaveActiveAnyTrue(value);
    }
    bool all_true(const bool value) {
        return WaveActiveAllTrue(value);
    }
    float prefix_sum(const float value) {
        return WavePrefixSum(value);
    }
    uint prefix_count(const bool value) {
        return WavePrefixCountBits(value);
    }
    uint first(const bool value) {
        return firstbitlow(WaveActiveBallot(value).x);
    }
    uint read(const uint value, const uint slot) {
        return WaveShuffle(value, slot);
    }
    float read(const float value, const uint slot) {
        return WaveShuffle(value, slot);
    }
    float3 read(const float3 value, const uint slot) {
        return WaveShuffle(value, slot);
    }
    bool read(const bool value, const uint slot) {
        return WaveShuffle(value, slot);
    }
    uint broadcast(const uint value, const uint slot) {
        return WaveReadLaneAt(value, slot);
    }
    float3 broadcast(const float3 value, const uint slot) {
        return WaveReadLaneAt(value, slot);
    }
}

groupshared float3 s_slot_values[LIGHT_GRID_SLOTS];

struct SharedSlots : ISlotGroup {
    uint slot;

    __init(const uint slot) {
        this.slot = slot;
    }

    float3 read(const float3 value, const uint from) {
        s_slot_values[slot] = value;
        GroupMemoryBarrierWithGroupSync();
        const float3 result = s_slot_values[from];
        GroupMemoryBarrierWithGroupSync();
        return result;
    }
    float read(const float value, const uint from) {
        return read(float3(value, 0.f, 0.f), from).x;
    }
    uint read(const uint value, const uint from) {
        return asuint(read(asfloat(value), from));
    }
    bool read(const bool value, const uint from) {
        return read(value ? 1u : 0u, from) != 0u;
    }
    uint broadcast(const uint value, const uint from) {
        return read(value, from);
    }
    float3 broadcast(const float3 value, const uint from) {
        return read(value, from);
    }

    float4 reduce(const float value) {
        s_slot_values[slot].x = value;
        GroupMemoryBarrierWithGroupSync();
        float4 result = float4(0.f, 0.f, -MERIAN_FLT_MAX, MERIAN_FLT_MAX);
        for (uint i = 0; i < LIGHT_GRID_SLOTS; i++) {
            const float v = s_slot_values[i].x;
            result.x += v;
            result.y += i < slot ? v : 0.f;
            result.z = max(result.z, v);
            result.w = min(result.w, v);
        }
        GroupMemoryBarrierWithGroupSync();
        return result;
    }
    float sum(const float value) {
        return reduce(value).x;
    }
    float maximum(const float value) {
        return reduce(value).z;
    }
    float minimum(const float value) {
        return reduce(value).w;
    }
    float prefix_sum(const float value) {
        return reduce(value).y;
    }
    uint count(const bool value) {
        return uint(reduce(value ? 1.f : 0.f).x);
    }
    uint prefix_count(const bool value) {
        return uint(reduce(value ? 1.f : 0.f).y);
    }
    bool any_true(const bool value) {
        return reduce(value ? 1.f : 0.f).z > 0.f;
    }
    bool all_true(const bool value) {
        return reduce(value ? 1.f : 0.f).w > 0.f;
    }
    uint first(const bool value) {
        const float4 r = reduce(value ? float(LIGHT_GRID_SLOTS - slot) : 0.f);
        return LIGHT_GRID_SLOTS - uint(r.z);
    }
}

// One past the last light of the slot's node.
uint cut_end<S : ISlotGroup>(const S slots, const uint first, const uint slot, const uint lights) {
    const uint next_first = slots.read(first, min(slot + 1u, LIGHT_GRID_SLOTS - 1u));
    return slot < LIGHT_GRID_SLOTS - 1u ? next_first : lights;
}

CutSlot cut_slot(const LightPreprocessBinding params,
                 const CellGeometry cg,
                 const uint first,
                 const uint end,
                 const Contribution contribution) {
    const TreeNode node = tree_node(params, first, end);
    return cut_slot(
        node, node_geometry(node, cg.center, cg.radius, nearest_distance(params, cg), cg.scale),
        contribution);
}

void store_cut<S : ISlotGroup>(const S slots,
                               const LightPreprocessBinding params,
                               const uint block,
                               const uint slot,
                               const uint key,
                               const uint first,
                               const Contribution contribution,
                               const CutSlot s,
                               const bool touched,
                               const float scale) {
    const uint lights = light_count(params);
    const bool listed = key != LIGHT_TREE_NO_KEY;
    const bool emits = listed && s.node.flux > 0.f;
    const float total = slots.sum(emits ? s.importance : 0.f);
    const float even = total > 0.f ? params.grid_even_share : 1.f;
    const uint emitting = slots.count(emits);
    const float p =
        emits ? (1.f - even) * (total > 0.f ? s.importance / total : 0.f) + even / float(emitting)
              : 0.f;

    const uint entry = block * LIGHT_GRID_SLOTS + slot;
    params.grid_keys[entry] = key;
    params.grid_contribution[entry] = contribution;
    params.grid_starts[entry] = listed ? first : lights;
    if (listed && first >= params.triangle_count && env_texels(params) > 0)
        params.grid_env_prefix[entry] = params.env.prefix(first - params.triangle_count);
    params.grid_estimate[entry] = s.geometry.x > 0.f ? scale / s.geometry.x : 1.f;
    if (touched) {
        for (uint i = 0; i < LIGHT_GRID_FEEDBACK_STRIDE; i++)
            params.grid_feedback[LIGHT_GRID_FEEDBACK_STRIDE * entry + i] = 0;
    }
    if (params.slot_weighing == LightSlotWeighing::LightSlotWeighingShadingPoint) {
        LightBound bound;
        bound.centroid = s.node.center;
        bound.flux = emits ? s.node.flux * max(relative_contribution(contribution, s.geometry).x,
                                               LIGHT_GRID_MIN_CONTRIBUTION)
                           : 0.f;
        bound.radius = s.node.radius;
        bound.flags = s.node.distant ? uint(EmissiveTriangleFlags::EmissiveDistant) : 0u;
        params.grid_slot_bounds[entry] = bound;
    } else {
        const float share = round(p * LIGHT_GRID_PROBABILITY_STEPS);
        params.grid_probability[entry] =
            (slots.prefix_sum(share) + share) * (1.f / LIGHT_GRID_PROBABILITY_STEPS);
    }
}

groupshared uint s_compact_key[LIGHT_GRID_SLOTS];
groupshared uint s_compact_first[LIGHT_GRID_SLOTS];
groupshared Contribution s_compact_contribution[LIGHT_GRID_SLOTS];

/* Each block holds the cut of a cell, or of the whole scene, through the light tree: a partition of
 * all lights into nodes, drawn by flux times their learned contribution, and by flux within a
 * node. A cut splits the node whose lights' contributions vary most and, once full, merges the pair
 * that varies least. Nodes are named by their first Morton code. (Walter et al. 2005; Vevoda et al.
 * 2018.)
 */
void grid_cell<S : ISlotGroup>(const S slots,
                               const LightPreprocessBinding params,
                               const uint block,
                               const uint slot) {
    const bool global = block == LIGHT_GRID_FALLBACK_BLOCK;
    const LightGridBlock cell_block = params.grid_blocks[block];
    const bool touched =
        params.grid_feedback[LIGHT_GRID_FEEDBACK_STRIDE * (block * LIGHT_GRID_SLOTS + slot)] != 0;
    const CellGeometry cg =
        global ? global_cell_geometry(params.grid_info) : block_geometry(params, cell_block);
    const uint lights = light_count(params);
    const LightGridInfo info = params.grid_info[cg.cascade];
    const uint last_slot = LIGHT_GRID_SLOTS - 1u;
    const uint next_slot = min(slot + 1u, last_slot);

    // carry the cut, or take the coarser cell's
    const bool lattice_kept =
        !params.grid_reset && same_lattice(params.tree_info[0], params.tree_info_prev[0]);
    const bool carried = lattice_kept && (global || cell_block.kept != 0);
    uint prev_cell = 0;
    const bool coarser =
        !global && !carried && lattice_kept &&
        coarser_cell(params.grid_info_prev, info, cg.center, cell_block.orientation, prev_cell);
    const uint coarser_block =
        coarser ? light_cell_block(params.grid_index_prev[prev_cell]) : LIGHT_GRID_FALLBACK_BLOCK;
    const bool inherited = coarser_block != LIGHT_GRID_FALLBACK_BLOCK;
    const uint prev_block = inherited ? coarser_block : block;

    uint key = LIGHT_TREE_NO_KEY;
    Contribution contribution = Contribution(0.f);
    float samples = 0.f;
    float2 feedback_sums = float2(0.f);
    if (carried || inherited) {
        const uint prev_entry = prev_block * LIGHT_GRID_SLOTS + slot;
        key = params.grid_keys_prev[prev_entry];
        contribution = params.grid_contribution_prev[prev_entry];
        if (carried) {
            const uint feedback_at = LIGHT_GRID_FEEDBACK_STRIDE * prev_entry;
            samples = float(params.grid_feedback_prev[feedback_at]);
            feedback_sums =
                float2(wide_sum(params.grid_feedback_prev, feedback_at + 1),
                       wide_sum(params.grid_feedback_prev, feedback_at + 3)) /
                float2(LIGHT_GRID_CONTRIBUTION_SCALE, LIGHT_GRID_CONTRIBUTION_SQUARE_SCALE);
        }
    }

    // a carried cut must still partition all lights
    bool fresh = !(carried || inherited);
    {
        const uint next_key = slots.read(key, next_slot);
        const bool listed = key != LIGHT_TREE_NO_KEY;
        const bool env = listed && key >= LIGHT_TREE_ENV_KEY;
        const uint prev_key = slots.read(key, slot > 0 ? slot - 1u : 0u);
        bool valid = slot != 0 || key == 0;
        if (slot < last_slot)
            valid &= listed ? next_key == LIGHT_TREE_NO_KEY || next_key > key
                            : next_key == LIGHT_TREE_NO_KEY;
        if (env) {
            valid &= key - LIGHT_TREE_ENV_KEY < env_texels(params);
            if (slot == 0 || prev_key < LIGHT_TREE_ENV_KEY)
                valid &= key == LIGHT_TREE_ENV_KEY;
        }
        if (!slots.all_true(valid) || slots.any_true(env) != (env_texels(params) > 0)) {
            key = slot == 0                              ? 0u
                  : slot == 1 && env_texels(params) > 0u ? LIGHT_TREE_ENV_KEY
                                                         : LIGHT_TREE_NO_KEY;
            fresh = true;
        }
    }

    uint first = node_start(params, key);
    const float nearest = nearest_distance(params, cg);
    const uint end = cut_end(slots, first, slot, lights);
    const TreeNode node = tree_node(params, first, key != LIGHT_TREE_NO_KEY ? end : first);
    const float3 geometry = node_geometry(node, cg.center, cg.radius, nearest, cg.scale);
    {
        if (fresh) {
            contribution = prior_contribution(geometry, prior_mean(params, node.distant));
        } else {
            if (global || inherited) {
                const LightGridInfo prev =
                    params.grid_info_prev[global ? params.grid_info_prev[0].cascade_count - 1u
                                                 : light_cell_cascade(prev_cell, info.dimension)];
                contribution = rescale_contribution(
                    contribution, cg.scale / light_cell_scale(prev.inv_cell_size));
            }
            if (inherited) {
                if (node.flux > 0.f)
                    contribution.z = min(contribution.z, LIGHT_GRID_PRIOR_SAMPLES);
            } else if (samples > 0.f && !node.is_empty()) {
                const float weight = 1.f / (samples + contribution.z);
                contribution.xy =
                    feedback_sums * weight * float2(geometry.x, geometry.x * geometry.x) +
                    contribution.xy * (contribution.z * weight);
                contribution.z = min(contribution.z + samples, LIGHT_GRID_CONTRIBUTION_MEMORY);
            }
        }
    }
    CutSlot s = cut_slot(node, geometry, contribution);
    bool evaluated = true;

    // merge dead and unlearned non-emitting nodes into their predecessor
    {
        const bool listed = key != LIGHT_TREE_NO_KEY;
        const float2 relative = relative_contribution(contribution, geometry);
        const bool dead = listed && !(relative.x > 0.f) && !(relative.y > 0.f) &&
                          contribution.z >= LIGHT_GRID_DEAD_SAMPLES;
        const bool previous_dead = slots.read(dead, slot > 0 ? slot - 1u : 0u);
        const uint previous_key = slots.read(key, slot > 0 ? slot - 1u : 0u);
        const bool absorbed = slot > 0 && listed &&
                              ((!(node.flux > 0.f) && contribution.z <= LIGHT_GRID_PRIOR_SAMPLES) ||
                               (dead && previous_dead)) &&
                              (key >= LIGHT_TREE_ENV_KEY) == (previous_key >= LIGHT_TREE_ENV_KEY);
        const bool keep = listed && !absorbed;
        if (slots.any_true(absorbed)) {
            const uint at = slots.prefix_count(keep);
            if (keep) {
                s_compact_key[at] = key;
                s_compact_first[at] = first;
                s_compact_contribution[at] = contribution;
            }
            GroupMemoryBarrierWithGroupSync();
            const uint kept = slots.count(keep);
            key = slot < kept ? s_compact_key[slot] : LIGHT_TREE_NO_KEY;
            first = slot < kept ? s_compact_first[slot] : lights;
            contribution = slot < kept ? s_compact_contribution[slot] : Contribution(0.f);
            evaluated = false;
        }
    }

    // refine
    const uint steps = fresh                           ? LIGHT_GRID_FRESH_SPLITS
                       : !carried                      ? LIGHT_GRID_SLOTS
                       : slots.any_true(samples > 0.f) ? params.grid_refinements
                                                       : 0u;
    for (uint step = 0; step < steps; step++) {
        const bool listed = key != LIGHT_TREE_NO_KEY;
        const uint end = cut_end(slots, first, slot, lights);
        if (!evaluated)
            s = cut_slot(params, cg, first, listed ? end : first, contribution);
        evaluated = true;

        const float candidate = s.node.splittable ? s.excess : 0.f;
        const float best = slots.maximum(candidate);
        if (!(best > 0.f))
            break;
        const uint split_slot = slots.first(candidate == best);

        const uint next_key = slots.read(key, next_slot);
        const uint next_end = slots.read(end, next_slot);
        const float next_excess = slots.read(s.excess, next_slot);
        const float next_flux = slots.read(s.node.flux, next_slot);
        const Contribution next_contribution = slots.read(contribution, next_slot);
        const bool full = !slots.any_true(!listed);
        const bool pair = full && slot < last_slot &&
                          (key >= LIGHT_TREE_ENV_KEY) == (next_key >= LIGHT_TREE_ENV_KEY);
        float merge_loss = 0.f;
        Contribution merged_contribution = contribution;
        if (pair) {
            const TreeNode merged_node = tree_node(params, first, next_end);
            const float3 merged_geometry =
                node_geometry(merged_node, cg.center, cg.radius, nearest, cg.scale);
            const float merged_flux = s.node.flux + next_flux;
            const float w = merged_flux > 0.f ? s.node.flux / merged_flux : 0.5f;
            const float merged_samples = w == 1.f   ? contribution.z
                                         : w == 0.f ? next_contribution.z
                                                    : min(contribution.z, next_contribution.z);
            merged_contribution = Contribution(
                w * contribution.xy + (1.f - w) * next_contribution.xy, merged_samples);
            const CutSlot merged = cut_slot(merged_node, merged_geometry, merged_contribution);
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
            const float prior = prior_mean(params, s.node.distant);
            low_contribution = halve_contribution(contribution, low.geometry, prior);
            high_contribution = halve_contribution(contribution, high.geometry, prior);
        }
        const float gain = 0.5f * best;
        split_first = slots.broadcast(split_first, split_slot);
        const uint split_key = slots.broadcast(s.node.split_key, split_slot);
        low_contribution = slots.broadcast(low_contribution, split_slot);
        high_contribution = slots.broadcast(high_contribution, split_slot);

        uint merge_slot = LIGHT_GRID_SLOTS;
        if (full) {
            const bool mergeable = pair && slot != split_slot && slot + 1u != split_slot;
            const float cheapest = slots.minimum(mergeable ? merge_loss : MERIAN_FLT_MAX);
            if (!slots.any_true(mergeable) || !(gain > LIGHT_GRID_REFINE_MARGIN * cheapest))
                break;
            merge_slot = slots.first(mergeable && merge_loss == cheapest);
        }

        uint target = split_slot;
        if (merge_slot < LIGHT_GRID_SLOTS) {
            const Contribution merged = slots.broadcast(merged_contribution, merge_slot);
            const uint from = slot > merge_slot ? next_slot : slot;
            key = slots.read(key, from);
            first = slots.read(first, from);
            contribution = slots.read(contribution, from);
            if (slot == last_slot) {
                key = LIGHT_TREE_NO_KEY;
                first = lights;
            }
            if (slot == merge_slot)
                contribution = merged;
            target -= target > merge_slot ? 1u : 0u;
        }

        const uint from = slot > target + 1u ? slot - 1u : slot;
        key = slots.read(key, from);
        first = slots.read(first, from);
        contribution = slots.read(contribution, from);
        if (slot == target)
            contribution = low_contribution;
        if (slot == target + 1u) {
            key = split_key;
            first = split_first;
            contribution = high_contribution;
        }
        evaluated = false;
    }

    if (!evaluated) {
        const uint end = cut_end(slots, first, slot, lights);
        s = cut_slot(params, cg, first, key != LIGHT_TREE_NO_KEY ? end : first, contribution);
    }
    store_cut(slots, params, block, slot, key, first, contribution, s, touched, cg.scale);
}

[shader("compute")][numthreads(LIGHT_GRID_SLOTS, 1, 1)] void
grid(uint3 group_id : SV_GroupID,
     uint slot : SV_GroupIndex,
     ParameterBlock<LightPreprocessBinding> params) {
    const uint block = group_id.x + group_id.y * LIGHT_DISPATCH_ROW;
    if (block >= params.block_count || (block != LIGHT_GRID_FALLBACK_BLOCK &&
                                        params.grid_blocks[block].idle == LIGHT_GRID_IDLE_FREE))
        return;
    if (WaveGetLaneCount() >= LIGHT_GRID_SLOTS)
        grid_cell(WaveSlots(), params, block, slot);
    else
        grid_cell(SharedSlots(slot), params, block, slot);
}
