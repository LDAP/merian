import merian_shaders.scene.light_data;

using merian;

static const uint CDF_GROUP_SIZE = 1024;

struct LightCdfBinding {
    StructuredBuffer<float> weights;
    RWStructuredBuffer<float> cdf;
    uint triangle_count;
};

groupshared float s_prefix[CDF_GROUP_SIZE];

// Inclusive prefix sum of the selection weights; a single workgroup walks the list in chunks.
[shader("compute")][numthreads(CDF_GROUP_SIZE, 1, 1)] void
main(uint thread_id : SV_GroupThreadID, ParameterBlock<LightCdfBinding> params) {
    float carry = 0.f;
    for (uint base = 0; base < params.triangle_count; base += CDF_GROUP_SIZE) {
        const uint i = base + thread_id;
        s_prefix[thread_id] = i < params.triangle_count ? params.weights[i] : 0.f;
        GroupMemoryBarrierWithGroupSync();

        // Hillis-Steele scan
        for (uint offset = 1; offset < CDF_GROUP_SIZE; offset <<= 1) {
            const float t = thread_id >= offset ? s_prefix[thread_id - offset] : 0.f;
            GroupMemoryBarrierWithGroupSync();
            s_prefix[thread_id] += t;
            GroupMemoryBarrierWithGroupSync();
        }

        if (i < params.triangle_count)
            params.cdf[i] = carry + s_prefix[thread_id];
        carry += s_prefix[CDF_GROUP_SIZE - 1];
        GroupMemoryBarrierWithGroupSync();
    }
}
