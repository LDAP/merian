import merian_shaders.colors.colorspaces;
import merian_shaders.utils.textures;

struct PushConstant {
    float firefly_percentile_lower;
    float firefly_percentile_upper;
    float adaptive_alpha_percentile_lower;
    float adaptive_alpha_percentile_upper;
};

[vk::push_constant]
ConstantBuffer<PushConstant> pc;

[vk::constant_id(0)]
const int workgroup_size_x = 1;
[vk::constant_id(1)]
const int workgroup_size_y = 1;

// graph io
Sampler2D<float4> in_src;

// percentile pass result, bound by the node
RWTexture2D<float4> quartiles;

// Rank into the sorted tile.
// ! building a uint4 from the spec-constant workgroup size lowers to an OpConstantComposite that fails Vulkan validation.
uint percentile_index(const float requested, const uint n) {
    return uint(clamp(round(requested * n), 0, n - 1));
}

//
// Percentile pass, quake mcpg. NVIDIA 5070 Laptop at 1080p (n=3); RX 7900 XTX (RADV) at wave64:
//
//                                     5070 Laptop      RX 7900 XTX
//   rank-select, packed key             0.1067 ms        0.0310 ms  (! not 100% accurate but acceptable for this)
//   rank-select, float compare          0.1550 ms        0.0451 ms
//   bitonic, wave-shuffle               0.2078 ms        0.0333 ms
//   bitonic, shared memory              0.3288 ms        0.0256 ms
//   odd-even sorted                     0.9900 ms        0.0350 ms
//
// ! For rank-select, packed key we delete 6 bits of the mantissa -> can pick a different value only among luminances
// within 2^-17 relative.

#define PERCENTILE_RANK_PACKED_KEY 0
#define PERCENTILE_RANK_EXACT 1
#define PERCENTILE_BITONIC_WAVE 2
#define PERCENTILE_BITONIC_SHARED 3

// Define PERCENTILE_STRATEGY to benchmark another one.
#ifndef PERCENTILE_STRATEGY
#ifdef MERIAN_TARGET_VENDOR_AMD
#define PERCENTILE_STRATEGY PERCENTILE_BITONIC_SHARED
#else
#define PERCENTILE_STRATEGY PERCENTILE_RANK_PACKED_KEY
#endif
#endif

#if PERCENTILE_STRATEGY == PERCENTILE_RANK_PACKED_KEY

groupshared uint key[workgroup_size_x * workgroup_size_y];
groupshared float percentile[4];

float4 compute_percentiles(const float own, const uint group_index, const uint n) {
    // Non-negative floats order the same as their bit patterns, so luminance and the thread index
    // pack into one sortable key. The index occupies the low mantissa bits, which makes the keys
    // unique and removes a tie-break, and ranking costs one integer compare per value.
    const uint own_key = (asuint(max(own, 0.0)) & ~(n - 1)) | group_index;
    key[group_index] = own_key;
    GroupMemoryBarrierWithGroupSync();

    uint rank = 0;
    for (uint i = 0; i < n; i++) {
        rank += uint(key[i] < own_key);
    }

    if (rank == percentile_index(pc.firefly_percentile_lower, n))
        percentile[0] = own;
    if (rank == percentile_index(pc.firefly_percentile_upper, n))
        percentile[1] = own;
    if (rank == percentile_index(pc.adaptive_alpha_percentile_lower, n))
        percentile[2] = own;
    if (rank == percentile_index(pc.adaptive_alpha_percentile_upper, n))
        percentile[3] = own;
    GroupMemoryBarrierWithGroupSync();

    return float4(percentile[0], percentile[1], percentile[2], percentile[3]);
}

#endif

#if PERCENTILE_STRATEGY == PERCENTILE_RANK_EXACT

groupshared float luminance[workgroup_size_x * workgroup_size_y];
groupshared float percentile[4];

float4 compute_percentiles(const float own, const uint group_index, const uint n) {
    luminance[group_index] = own;
    if (group_index < 4)
        percentile[group_index] = 0.0; // a NaN input leaves its rank unclaimed
    GroupMemoryBarrierWithGroupSync();

    // Only four order statistics are needed, so each thread ranks its own value rather than the
    // group sorting itself. The index tie-break makes the ranks a permutation.
    uint rank = 0;
    for (uint i = 0; i < n; i++) {
        const float other = luminance[i];
        rank += uint(other < own || (other == own && i < group_index));
    }

    if (rank == percentile_index(pc.firefly_percentile_lower, n))
        percentile[0] = own;
    if (rank == percentile_index(pc.firefly_percentile_upper, n))
        percentile[1] = own;
    if (rank == percentile_index(pc.adaptive_alpha_percentile_lower, n))
        percentile[2] = own;
    if (rank == percentile_index(pc.adaptive_alpha_percentile_upper, n))
        percentile[3] = own;
    GroupMemoryBarrierWithGroupSync();

    return float4(percentile[0], percentile[1], percentile[2], percentile[3]);
}

#endif

#if PERCENTILE_STRATEGY == PERCENTILE_BITONIC_WAVE

groupshared float exchange[workgroup_size_x * workgroup_size_y];
groupshared float percentile[4];
groupshared uint wave_counter;

float4 compute_percentiles(const float own_in, const uint group_index, const uint n) {
    // Which sorted position a thread ends up holding is arbitrary, so take it from the wave and
    // lane the thread runs on rather than from the group index.
    if (group_index == 0)
        wave_counter = 0;
    GroupMemoryBarrierWithGroupSync();

    uint wave = 0;
    if (WaveIsFirstLane())
        InterlockedAdd(wave_counter, 1, wave);
    wave = WaveReadLaneFirst(wave);

    const uint lane = WaveGetLaneIndex();
    const uint lane_count = WaveGetLaneCount();
    const uint position = wave * lane_count + lane;

    // Only strides at or above the wave width need shared memory and a barrier (one of the 21
    // stages at wave32, none at wave64).
    float own = own_in;
    for (uint k = 2; k <= n; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float other;
            if (j >= lane_count) {
                GroupMemoryBarrierWithGroupSync();
                exchange[position] = own;
                GroupMemoryBarrierWithGroupSync();
                other = exchange[position ^ j];
            } else {
                other = WaveShuffle(own, lane ^ j);
            }
            const bool ascending = (position & k) == 0;
            const bool keep_min = ((position & j) == 0) == ascending;
            own = keep_min ? min(own, other) : max(own, other);
        }
    }

    // Each thread now holds the value at its own sorted position.
    if (position == percentile_index(pc.firefly_percentile_lower, n))
        percentile[0] = own;
    if (position == percentile_index(pc.firefly_percentile_upper, n))
        percentile[1] = own;
    if (position == percentile_index(pc.adaptive_alpha_percentile_lower, n))
        percentile[2] = own;
    if (position == percentile_index(pc.adaptive_alpha_percentile_upper, n))
        percentile[3] = own;
    GroupMemoryBarrierWithGroupSync();

    return float4(percentile[0], percentile[1], percentile[2], percentile[3]);
}

#endif

#if PERCENTILE_STRATEGY == PERCENTILE_BITONIC_SHARED

groupshared float sort[workgroup_size_x * workgroup_size_y];

float4 compute_percentiles(const float own, const uint group_index, const uint n) {
    sort[group_index] = own;
    GroupMemoryBarrierWithGroupSync();

    // n must be a power of two. Only the lower thread of each compared pair (ixj > group_index)
    // writes both slots, so a stage has no intra-slot race.
    for (uint k = 2; k <= n; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint ixj = group_index ^ j;
            if (ixj > group_index) {
                const float a = sort[group_index];
                const float b = sort[ixj];
                const bool ascending = (group_index & k) == 0;
                if ((a > b) == ascending) {
                    sort[group_index] = b;
                    sort[ixj] = a;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    return float4(sort[percentile_index(pc.firefly_percentile_lower, n)],
                  sort[percentile_index(pc.firefly_percentile_upper, n)],
                  sort[percentile_index(pc.adaptive_alpha_percentile_lower, n)],
                  sort[percentile_index(pc.adaptive_alpha_percentile_upper, n)]);
}

#endif

[numthreads(workgroup_size_x, workgroup_size_y, 1)]
[shader("compute")]
void main(uint3 gl_WorkGroupID: SV_GroupID,
          uint3 gl_GlobalInvocationID: SV_DispatchThreadID,
          uint group_index: SV_GroupIndex) {

    const uint n = uint(workgroup_size_x * workgroup_size_y);
    const int2 pixel = min(gl_GlobalInvocationID.xy, int2(merian::texture_dimensions(in_src)) - 1);

    const float own = merian::yuv_luminance(in_src[pixel].rgb);
    const float4 percentiles = compute_percentiles(own, group_index, n);

    if (group_index == 0) {
        quartiles.Store(gl_WorkGroupID.xy, percentiles);
    }
}
