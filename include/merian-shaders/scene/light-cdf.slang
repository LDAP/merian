import merian_shaders.scene.light_data;

using merian;

static const uint CDF_GROUP_SIZE = 1024;
static const uint CDF_PUBLISHED = 1;

struct LightCdfBinding {
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    StructuredBuffer<uint> order;
    RWStructuredBuffer<float2> cdf;
    // zeroed: the blocks started, then per block a flag and its total
    RWStructuredBuffer<uint> state;
    uint count;
};

struct LightGuideBinding {
    StructuredBuffer<float2> cdf;
    RWStructuredBuffer<uint> guide;
    uint count;
    uint buckets;
};

void write_guide(const StructuredBuffer<float2> cdf,
                 const RWStructuredBuffer<uint> guide,
                 const uint count,
                 const uint buckets,
                 const uint bucket) {
    if (bucket > buckets)
        return;
    const float total = count > 0 ? cdf[count - 1].x : 0.f;
    const float scale = total > 0.f ? float(buckets) / total : 0.f;
    if (bucket == 0)
        guide[buckets + 1] = asuint(scale);
    uint lo = 0;
    uint hi = count;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (cdf_bucket(cdf[mid].x, scale, buckets) >= bucket)
            hi = mid;
        else
            lo = mid + 1;
    }
    guide[bucket] = lo;
}

[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
guide(uint bucket : SV_DispatchThreadID, ParameterBlock<LightGuideBinding> params) {
    write_guide(params.cdf, params.guide, params.count, params.buckets, bucket);
}

// One entry per wave; a wave size of one would mean one per thread.
groupshared float2 s_wave_total[CDF_GROUP_SIZE];
groupshared float2 s_wave_offset[CDF_GROUP_SIZE];
groupshared float2 s_group_total;
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
    const uint lane = WaveGetLaneIndex();

    const float2 wave_prefix = wave_scan(value);
    if (lane == lane_count - 1)
        s_wave_total[wave] = wave_prefix;
    GroupMemoryBarrierWithGroupSync();

    if (wave == 0) {
        const uint wave_count = CDF_GROUP_SIZE / lane_count;
        float2 carry = float2(0);
        for (uint first = 0; first < wave_count; first += lane_count) {
            const uint w = first + lane;
            const float2 inclusive =
                ff_add(carry, wave_scan(w < wave_count ? s_wave_total[w] : float2(0)));
            const float2 previous = WaveShuffle(inclusive, lane > 0 ? lane - 1 : 0);
            if (w < wave_count)
                s_wave_offset[w] = lane > 0 ? previous : carry;
            carry = WaveReadLaneAt(inclusive, lane_count - 1);
        }
        if (lane == 0)
            s_group_total = carry;
    }
    GroupMemoryBarrierWithGroupSync();

    total = s_group_total;
    return ff_add(s_wave_offset[wave], wave_prefix);
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
    const float flux = i < params.count ? params.proxies[params.order[i]].flux : 0.f;

    float2 total;
    const float2 prefix = group_scan(float2(max(flux, 0.f), 0.f), thread_id, total);
    const float2 before = block_prefix(params.state, block, total, thread_id);
    if (i < params.count)
        params.cdf[i] = ff_add(before, prefix);
}
