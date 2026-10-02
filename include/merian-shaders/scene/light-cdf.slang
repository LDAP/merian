import merian_shaders.scene.light_data;

using merian;

static const uint CDF_GROUP_SIZE = 1024;

struct LightCdfBinding {
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    StructuredBuffer<uint> order;
    StructuredBuffer<float> values;
    bool gather;
    RWStructuredBuffer<float2> cdf;
    // one entry per block, scanned in place into the offsets the blocks are shifted by
    RWStructuredBuffer<float2> block_sums;
    uint count;
    uint block_count;
};

// One entry per wave; a wave size of one would mean one per thread.
groupshared float2 s_wave_total[CDF_GROUP_SIZE];

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

// Inclusive prefix sum, as three passes.
[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
scan_blocks(uint3 group_id: SV_GroupID,
            uint thread_id: SV_GroupIndex,
            ParameterBlock<LightCdfBinding> params) {
    const uint i = group_id.x * CDF_GROUP_SIZE + thread_id;
    float value = 0.f;
    if (i < params.count)
        value = params.gather ? params.proxies[params.order[i]].flux : params.values[i];

    float2 total;
    const float2 prefix = group_scan(float2(max(value, 0.f), 0.f), thread_id, total);

    if (i < params.count)
        params.cdf[i] = prefix;
    if (thread_id == 0)
        params.block_sums[group_id.x] = total;
}

// The block totals are few, so one group walking them in chunks costs nothing.
[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
scan_block_sums(uint thread_id: SV_GroupIndex, ParameterBlock<LightCdfBinding> params) {
    float2 carry = float2(0);
    for (uint base = 0; base < params.block_count; base += CDF_GROUP_SIZE) {
        const uint i = base + thread_id;
        const float2 value = i < params.block_count ? params.block_sums[i] : float2(0);

        float2 total;
        const float2 prefix = group_scan(value, thread_id, total);

        if (i < params.block_count)
            params.block_sums[i] = ff_add(carry, prefix);
        carry = ff_add(carry, total);
        GroupMemoryBarrierWithGroupSync();
    }
}

// Shifts every block past the first by the totals of the blocks before it.
[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
add_block_offset(uint3 group_id: SV_GroupID,
                 uint thread_id: SV_GroupIndex,
                 ParameterBlock<LightCdfBinding> params) {
    if (group_id.x == 0)
        return;
    const uint i = group_id.x * CDF_GROUP_SIZE + thread_id;
    if (i >= params.count)
        return;
    params.cdf[i] = ff_add(params.block_sums[group_id.x - 1], params.cdf[i]);
}
