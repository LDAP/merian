import merian_shaders.scene.light_data;

using merian;

static const uint CDF_GROUP_SIZE = 1024;
static const uint CDF_PUBLISHED = 1;

struct LightCdfBinding {
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    StructuredBuffer<uint> order;
    // the fourth power of the radius instead of the flux
    bool extent;
    RWStructuredBuffer<float2> cdf;
    // zeroed: the blocks started, then per block a flag and its total
    RWStructuredBuffer<uint> state;
    uint count;
};

// One entry per wave; a wave size of one would mean one per thread.
groupshared float2 s_wave_total[CDF_GROUP_SIZE];
groupshared uint s_block;

float2 wave_scan(float2 value) {
    const uint lane = WaveGetLaneIndex();
    for (uint offset = 1; offset < WaveGetLaneCount(); offset <<= 1) {
        const float2 other = WaveShuffle(value, lane >= offset ? lane - offset : lane);
        if (lane >= offset)
            value = ff_add(other, value);
    }
    return value;
}

// Inclusive scan across the group, per wave and then over the wave totals. Every thread has to
// reach it: it synchronizes the group.
float2 group_scan(const float2 value, const uint thread_id, out float2 total) {
    const uint lane_count = WaveGetLaneCount();
    const uint wave = thread_id / lane_count;

    const float2 wave_prefix = wave_scan(value);
    if (WaveGetLaneIndex() == lane_count - 1)
        s_wave_total[wave] = wave_prefix;
    GroupMemoryBarrierWithGroupSync();

    const uint wave_count = CDF_GROUP_SIZE / lane_count;
    float2 offset = float2(0);
    total = float2(0);
    for (uint w = 0; w < wave_count; w++) {
        if (w == wave)
            offset = total;
        total = ff_add(total, s_wave_total[w]);
    }
    return ff_add(offset, wave_prefix);
}

// Blocks in launch order, so every block one waits for has started.
uint start_block(const RWStructuredBuffer<uint> state, const uint thread_id) {
    if (thread_id == 0) {
        uint started;
        InterlockedAdd(state[0], 1u, started);
        s_block = started;
    }
    GroupMemoryBarrierWithGroupSync();
    return s_block;
}

uint block_status_at(const uint block) {
    return 1u + 3u * block;
}

void publish_total(const RWStructuredBuffer<uint> state, const uint block, const float2 total) {
    const uint at = block_status_at(block);
    uint previous;
    InterlockedExchange(state[at + 1u], asuint(total.x), previous);
    InterlockedExchange(state[at + 2u], asuint(total.y), previous);
    AllMemoryBarrier();
    InterlockedExchange(state[at], CDF_PUBLISHED, previous);
}

float2 published_total(const RWStructuredBuffer<uint> state, const uint block) {
    const uint at = block_status_at(block);
    uint status = 0;
    while (status == 0)
        InterlockedOr(state[at], 0u, status);
    AllMemoryBarrier();
    uint hi;
    uint lo;
    InterlockedOr(state[at + 1u], 0u, hi);
    InterlockedOr(state[at + 2u], 0u, lo);
    return float2(asfloat(hi), asfloat(lo));
}

// The sum of the blocks before this one, from the totals they publish (decoupled look-back,
// Merrill and Garland 2016), added in a fixed order so that every run sums alike.
float2 block_prefix(const RWStructuredBuffer<uint> state,
                    const uint block,
                    const float2 total,
                    const uint thread_id) {
    if (thread_id == 0)
        publish_total(state, block, total);
    float2 before = float2(0);
    for (uint first = 0; first < block; first += CDF_GROUP_SIZE) {
        const uint earlier = first + thread_id;
        const float2 value = earlier < block ? published_total(state, earlier) : float2(0);
        GroupMemoryBarrierWithGroupSync();
        float2 chunk;
        group_scan(value, thread_id, chunk);
        before = ff_add(before, chunk);
    }
    return before;
}

// Inclusive prefix sum of the flux in tree order, in one pass.
[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
scan(uint thread_id : SV_GroupIndex, ParameterBlock<LightCdfBinding> params) {
    const uint block = start_block(params.state, thread_id);
    const uint i = block * CDF_GROUP_SIZE + thread_id;
    float value = 0.f;
    if (i < params.count) {
        const EmissiveLightProxy proxy = params.proxies[params.order[i]];
        value =
            params.extent ? proxy.radius * proxy.radius * proxy.radius * proxy.radius : proxy.flux;
    }

    float2 total;
    const float2 prefix = group_scan(float2(max(value, 0.f), 0.f), thread_id, total);
    const float2 before = block_prefix(params.state, block, total, thread_id);
    if (i < params.count)
        params.cdf[i] = ff_add(before, prefix);
}
