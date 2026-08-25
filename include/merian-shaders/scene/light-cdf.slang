import merian_shaders.scene.light_data;

using merian;

static const uint CDF_GROUP_SIZE = 1024;

struct LightCdfBinding {
    // the flux alone is needed here, and the proxy is half the record's bytes
    StructuredBuffer<EmissiveLightProxy, ScalarDataLayout> proxies;
    RWStructuredBuffer<float> cdf;
    // one entry per block, scanned in place into the offsets the blocks are shifted by
    RWStructuredBuffer<float> block_sums;
    uint triangle_count;
    uint block_count;
};

// One entry per wave; a wave size of one would mean one per thread.
groupshared float s_wave_total[CDF_GROUP_SIZE];

// Inclusive scan across the group, per wave and then over the wave totals. Every thread has to
// reach it: it synchronizes the group.
float group_scan(const float value, const uint thread_id, out float total) {
    const uint lane_count = WaveGetLaneCount();
    const uint wave = thread_id / lane_count;

    const float wave_prefix = WavePrefixSum(value) + value;
    const float wave_sum = WaveActiveSum(value);
    if (WaveIsFirstLane())
        s_wave_total[wave] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    const uint wave_count = CDF_GROUP_SIZE / lane_count;
    float offset = 0.f;
    total = 0.f;
    for (uint w = 0; w < wave_count; w++) {
        if (w == wave)
            offset = total;
        total += s_wave_total[w];
    }
    return offset + wave_prefix;
}

/* Inclusive prefix sum of the flux, as three passes.
 *
 * A single group walking the list in chunks is bound by the load latency of one chunk after
 * another, whatever the scan inside it costs; with a light count in the hundred thousands that is
 * most of the light preprocess. Scanning the blocks independently puts the whole list in flight at
 * once, and only the scan over the block totals stays serial.
 */
[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
scan_blocks(uint3 group_id: SV_GroupID,
            uint thread_id: SV_GroupIndex,
            ParameterBlock<LightCdfBinding> params) {
    const uint i = group_id.x * CDF_GROUP_SIZE + thread_id;
    const float flux = i < params.triangle_count ? params.proxies[i].flux : 0.f;

    float total;
    const float prefix = group_scan(flux, thread_id, total);

    if (i < params.triangle_count)
        params.cdf[i] = prefix;
    if (thread_id == 0)
        params.block_sums[group_id.x] = total;
}

// The block totals are few, so one group walking them in chunks costs nothing.
[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
scan_block_sums(uint thread_id: SV_GroupIndex, ParameterBlock<LightCdfBinding> params) {
    float carry = 0.f;
    for (uint base = 0; base < params.block_count; base += CDF_GROUP_SIZE) {
        const uint i = base + thread_id;
        const float value = i < params.block_count ? params.block_sums[i] : 0.f;

        float total;
        const float prefix = group_scan(value, thread_id, total);

        if (i < params.block_count)
            params.block_sums[i] = carry + prefix;
        carry += total;
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
    if (i >= params.triangle_count)
        return;
    params.cdf[i] += params.block_sums[group_id.x - 1];
}
