import merian_shaders.scene.light_data;
import merian_shaders.utils.morton;

using merian;

// Least significant digit radix sort of the light tree keys: count, scan and a stable scatter per
// digit.

static const uint SORT_RADIX = 1u << LIGHT_SORT_RADIX_BITS;
static const uint SORT_ITEMS = LIGHT_SORT_TILE / LIGHT_SORT_GROUP;
static const uint SORT_SCAN_GROUP = 1024;
static const uint SORT_MAX_WAVES = LIGHT_SORT_GROUP / 16;

struct LightSortBinding {
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    StructuredBuffer<LightTreeInfo, ScalarDataLayout> tree_info;
    // the first pass writes them
    RWStructuredBuffer<uint> keys_in;
    StructuredBuffer<uint> values_in;
    RWStructuredBuffer<uint> keys_out;
    RWStructuredBuffer<uint> values_out;
    // digit-major
    RWStructuredBuffer<uint> histogram;
    RWStructuredBuffer<uint> rank;
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

groupshared uint s_histogram[SORT_RADIX];

[shader("compute")][numthreads(LIGHT_SORT_GROUP, 1, 1)] void
sort_count(uint3 group_id : SV_GroupID,
           uint thread_id : SV_GroupIndex,
           ParameterBlock<LightSortBinding> params) {
    for (uint d = thread_id; d < SORT_RADIX; d += LIGHT_SORT_GROUP)
        s_histogram[d] = 0;
    GroupMemoryBarrierWithGroupSync();

    const uint base = group_id.x * LIGHT_SORT_TILE;
    for (uint item = 0; item < SORT_ITEMS; item++) {
        const uint i = base + item * LIGHT_SORT_GROUP + thread_id;
        if (i >= params.count)
            break;
        const uint key = params.pass == 0
                             ? light_tree_key(params.proxies[i].centroid, params.tree_info[0])
                             : params.keys_in[i];
        if (params.pass == 0)
            params.keys_in[i] = key;
        InterlockedAdd(s_histogram[digit_of(key, params.pass)], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint d = thread_id; d < SORT_RADIX; d += LIGHT_SORT_GROUP)
        params.histogram[d * params.tile_count + group_id.x] = s_histogram[d];
}

groupshared uint s_scan_waves[SORT_SCAN_GROUP];

[shader("compute")][numthreads(SORT_SCAN_GROUP, 1, 1)] void
sort_scan(uint thread_id : SV_GroupIndex, ParameterBlock<LightSortBinding> params) {
    const uint lane_count = WaveGetLaneCount();
    const uint wave = thread_id / lane_count;
    const uint entries = SORT_RADIX * params.tile_count;
    uint carry = 0;
    for (uint base = 0; base < entries; base += SORT_SCAN_GROUP) {
        const uint i = base + thread_id;
        const uint value = i < entries ? params.histogram[i] : 0u;
        const uint wave_prefix = WavePrefixSum(value);
        const uint wave_total = WaveActiveSum(value);
        if (WaveIsFirstLane())
            s_scan_waves[wave] = wave_total;
        GroupMemoryBarrierWithGroupSync();

        uint offset = 0;
        uint total = 0;
        for (uint w = 0; w < SORT_SCAN_GROUP / lane_count; w++) {
            if (w == wave)
                offset = total;
            total += s_scan_waves[w];
        }
        if (i < entries)
            params.histogram[i] = carry + offset + wave_prefix;
        carry += total;
        GroupMemoryBarrierWithGroupSync();
    }
}

groupshared uint s_digit_offset[SORT_RADIX];
groupshared uint s_wave_digits[SORT_MAX_WAVES * SORT_RADIX];

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

[shader("compute")][numthreads(LIGHT_SORT_GROUP, 1, 1)] void
sort_scatter(uint3 group_id : SV_GroupID,
             uint thread_id : SV_GroupIndex,
             ParameterBlock<LightSortBinding> params) {
    const uint lane_count = WaveGetLaneCount();
    const uint wave_count = LIGHT_SORT_GROUP / lane_count;
    const uint wave = thread_id / lane_count;
    const uint lane = WaveGetLaneIndex();
    const bool last_pass = params.pass + 1 == LIGHT_SORT_PASSES;

    for (uint d = thread_id; d < SORT_RADIX; d += LIGHT_SORT_GROUP) {
        s_digit_offset[d] = params.histogram[d * params.tile_count + group_id.x];
        for (uint w = 0; w < wave_count; w++)
            s_wave_digits[w * SORT_RADIX + d] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint base = group_id.x * LIGHT_SORT_TILE;
    for (uint item = 0; item < SORT_ITEMS; item++) {
        const uint i = base + item * LIGHT_SORT_GROUP + thread_id;
        const bool valid = i < params.count;
        const uint key = valid ? params.keys_in[i] : 0u;
        const uint value = !valid ? 0u : params.pass == 0 ? i : params.values_in[i];
        const uint digit = digit_of(key, params.pass);

        uint4 peers = WaveActiveBallot(valid);
        for (uint b = 0; b < LIGHT_SORT_RADIX_BITS; b++) {
            const bool set = ((digit >> b) & 1u) != 0;
            const uint4 ballot = WaveActiveBallot(set);
            peers &= set ? ballot : ~ballot;
        }
        const uint below = count_below(peers, lane);
        const uint same =
            countbits(peers.x) + countbits(peers.y) + countbits(peers.z) + countbits(peers.w);
        if (valid && below + 1 == same)
            s_wave_digits[wave * SORT_RADIX + digit] = same;
        GroupMemoryBarrierWithGroupSync();

        if (valid) {
            uint at = s_digit_offset[digit] + below;
            for (uint w = 0; w < wave; w++)
                at += s_wave_digits[w * SORT_RADIX + digit];
            params.keys_out[at] = key;
            params.values_out[at] = value;
            if (last_pass)
                params.rank[value] = at;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint d = thread_id; d < SORT_RADIX; d += LIGHT_SORT_GROUP) {
            uint round = 0;
            for (uint w = 0; w < wave_count; w++) {
                round += s_wave_digits[w * SORT_RADIX + d];
                s_wave_digits[w * SORT_RADIX + d] = 0;
            }
            s_digit_offset[d] += round;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
