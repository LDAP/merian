import merian_shaders.scene.light_data;
import merian_shaders.utils.morton;

using merian;

// Least significant digit radix sort of the light tree keys, one pass per digit over tiles that
// find their offsets by decoupled look-back (Merrill and Garland 2016; Adinets and Merrill 2022).

static const uint SORT_RADIX = 1u << LIGHT_SORT_RADIX_BITS;
static const uint SORT_ITEMS = LIGHT_SORT_TILE / LIGHT_SORT_GROUP;
static const uint SORT_MAX_WAVES = LIGHT_SORT_GROUP / 16;
static const uint SORT_FLAG_AGGREGATE = 1u << 30;
static const uint SORT_FLAG_PREFIX = 2u << 30;
static const uint SORT_COUNT_MASK = SORT_FLAG_AGGREGATE - 1u;

struct LightSortBinding {
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    StructuredBuffer<LightTreeInfo, ScalarDataLayout> tree_info;
    // the histogram pass writes them
    RWStructuredBuffer<uint> keys_in;
    StructuredBuffer<uint> values_in;
    RWStructuredBuffer<uint> keys_out;
    RWStructuredBuffer<uint> values_out;
    RWStructuredBuffer<uint> rank;
    // zeroed: every pass's digit counts, every pass's tiles started, then the tiles' digit status
    RWStructuredBuffer<uint> state;
    uint count;
    uint tile_count;
    uint pass;
};

uint light_tree_key(const float3 pos, const LightTreeInfo info) {
    const float3 cell = (pos - info.origin) * info.inv_cell_size;
    if (!all(isfinite(cell)))
        return 0;
    const float last = float((1u << LIGHT_TREE_SIDE_BITS) - 1u);
    return morton_encode3d(uint3(min(max(cell, float3(0.f)), float3(last))));
}

uint digit_of(const uint key, const uint pass) {
    return (key >> (pass * LIGHT_SORT_RADIX_BITS)) & (SORT_RADIX - 1u);
}

uint histogram_at(const uint pass, const uint digit) {
    return pass * SORT_RADIX + digit;
}

uint tiles_started_at(const uint pass) {
    return LIGHT_SORT_PASSES * SORT_RADIX + pass;
}

uint status_at(const LightSortBinding params, const uint tile, const uint digit) {
    return LIGHT_SORT_PASSES * (SORT_RADIX + 1u) +
           (params.pass * params.tile_count + tile) * SORT_RADIX + digit;
}

groupshared uint s_histogram[LIGHT_SORT_PASSES * SORT_RADIX];

// The digit counts of every pass at once: they do not depend on the order.
[shader("compute")][numthreads(LIGHT_SORT_GROUP, 1, 1)] void
sort_histogram(uint3 group_id : SV_GroupID,
               uint thread_id : SV_GroupIndex,
               ParameterBlock<LightSortBinding> params) {
    for (uint d = thread_id; d < LIGHT_SORT_PASSES * SORT_RADIX; d += LIGHT_SORT_GROUP)
        s_histogram[d] = 0;
    GroupMemoryBarrierWithGroupSync();

    const LightTreeInfo info = params.tree_info[0];
    const uint base = group_id.x * LIGHT_SORT_TILE;
    for (uint item = 0; item < SORT_ITEMS; item++) {
        const uint i = base + item * LIGHT_SORT_GROUP + thread_id;
        if (i >= params.count)
            break;
        const uint key = light_tree_key(params.proxies[i].centroid, info);
        params.keys_in[i] = key;
        for (uint pass = 0; pass < LIGHT_SORT_PASSES; pass++)
            InterlockedAdd(s_histogram[histogram_at(pass, digit_of(key, pass))], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint d = thread_id; d < LIGHT_SORT_PASSES * SORT_RADIX; d += LIGHT_SORT_GROUP) {
        if (s_histogram[d] > 0)
            InterlockedAdd(params.state[d], s_histogram[d]);
    }
}

groupshared uint s_tile;
groupshared uint s_digit_offset[SORT_RADIX];
// per item, wave and digit: the keys of the digit, then the tile's keys of the digit before them
groupshared uint s_round_digits[SORT_ITEMS * SORT_MAX_WAVES * SORT_RADIX];
groupshared uint s_wave_sums[SORT_MAX_WAVES];

uint count_below(const uint4 mask, const uint lane) {
    uint below = 0;
    for (uint c = 0; c < 4; c++) {
        const uint first = 32u * c;
        const uint lt = lane >= first + 32u ? 0xFFFFFFFFu
                        : lane > first      ? (1u << (lane - first)) - 1u
                                            : 0u;
        below += countbits(mask[c] & lt);
    }
    return below;
}

void publish(const LightSortBinding params, const uint tile, const uint digit, const uint status) {
    uint previous;
    InterlockedExchange(params.state[status_at(params, tile, digit)], status, previous);
}

// The digit's keys in the tiles before this one, from the counts the tiles publish.
uint look_back(const LightSortBinding params, const uint tile, const uint digit, const uint local) {
    if (tile == 0) {
        publish(params, tile, digit, SORT_FLAG_PREFIX | local);
        return 0;
    }
    publish(params, tile, digit, SORT_FLAG_AGGREGATE | local);
    uint before = 0;
    uint t = tile;
    while (t > 0) {
        uint status;
        InterlockedOr(params.state[status_at(params, t - 1u, digit)], 0u, status);
        if ((status & ~SORT_COUNT_MASK) == 0)
            continue;
        before += status & SORT_COUNT_MASK;
        if ((status & SORT_FLAG_PREFIX) != 0)
            break;
        t--;
    }
    publish(params, tile, digit, SORT_FLAG_PREFIX | (before + local));
    return before;
}

// One thread per digit.
[shader("compute")][numthreads(LIGHT_SORT_GROUP, 1, 1)] void
sort_scatter(uint thread_id : SV_GroupIndex, ParameterBlock<LightSortBinding> params) {
    const uint lane_count = WaveGetLaneCount();
    const uint wave_count = LIGHT_SORT_GROUP / lane_count;
    const uint wave = thread_id / lane_count;
    const uint lane = WaveGetLaneIndex();
    const bool last_pass = params.pass + 1 == LIGHT_SORT_PASSES;

    // tiles in launch order, so every tile one waits for has started
    if (thread_id == 0) {
        uint started;
        InterlockedAdd(params.state[tiles_started_at(params.pass)], 1u, started);
        s_tile = started;
    }
    for (uint k = thread_id; k < SORT_ITEMS * wave_count * SORT_RADIX; k += LIGHT_SORT_GROUP)
        s_round_digits[k] = 0;
    GroupMemoryBarrierWithGroupSync();

    // rank every key among the keys of its digit in its wave and item
    const uint tile = s_tile;
    const uint base = tile * LIGHT_SORT_TILE;
    uint keys[SORT_ITEMS];
    uint values[SORT_ITEMS];
    uint below[SORT_ITEMS];
    for (uint item = 0; item < SORT_ITEMS; item++) {
        const uint i = base + item * LIGHT_SORT_GROUP + thread_id;
        const bool valid = i < params.count;
        keys[item] = valid ? params.keys_in[i] : 0u;
        values[item] = !valid ? 0u : params.pass == 0 ? i : params.values_in[i];
        const uint digit = digit_of(keys[item], params.pass);

        uint4 peers = WaveActiveBallot(valid);
        for (uint b = 0; b < LIGHT_SORT_RADIX_BITS; b++) {
            const bool set = ((digit >> b) & 1u) != 0;
            const uint4 ballot = WaveActiveBallot(set);
            peers &= set ? ballot : ~ballot;
        }
        below[item] = count_below(peers, lane);
        const uint same =
            countbits(peers.x) + countbits(peers.y) + countbits(peers.z) + countbits(peers.w);
        if (valid && below[item] + 1 == same)
            s_round_digits[(item * wave_count + wave) * SORT_RADIX + digit] = same;
    }
    GroupMemoryBarrierWithGroupSync();

    // the digit's keys in item and wave order, in earlier tiles and of smaller digits
    uint local = 0;
    for (uint item = 0; item < SORT_ITEMS; item++) {
        for (uint w = 0; w < wave_count; w++) {
            const uint at = (item * wave_count + w) * SORT_RADIX + thread_id;
            const uint round = s_round_digits[at];
            s_round_digits[at] = local;
            local += round;
        }
    }
    const uint digit_count = params.state[histogram_at(params.pass, thread_id)];
    const uint wave_prefix = WavePrefixSum(digit_count);
    if (lane == lane_count - 1u)
        s_wave_sums[wave] = wave_prefix + digit_count;
    const uint before = look_back(params, tile, thread_id, local);
    GroupMemoryBarrierWithGroupSync();
    uint smaller = wave_prefix;
    for (uint w = 0; w < wave; w++)
        smaller += s_wave_sums[w];
    s_digit_offset[thread_id] = smaller + before;
    GroupMemoryBarrierWithGroupSync();

    for (uint item = 0; item < SORT_ITEMS; item++) {
        const uint i = base + item * LIGHT_SORT_GROUP + thread_id;
        if (i >= params.count)
            continue;
        const uint digit = digit_of(keys[item], params.pass);
        const uint at = s_digit_offset[digit] +
                        s_round_digits[(item * wave_count + wave) * SORT_RADIX + digit] +
                        below[item];
        params.keys_out[at] = keys[item];
        params.values_out[at] = values[item];
        if (last_pass)
            params.rank[values[item]] = at;
    }
}
