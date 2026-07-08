// Thin merian wrapper around the NVIDIA SHARC world-space radiance cache.
// All SHARC-typed code is confined to the Sharc struct so importers only touch merian types.
// https://github.com/NVIDIA-RTX/SHARC

#define SHARC_UPDATE 0
#define SHARC_ENABLE_RESPONSIVE_LIGHTING 0
#define SHARC_ENABLE_SH_ENCODING 0
#define SHARC_ENABLE_GLSL 0
#define HASH_GRID_COMPACT 0
#define HASH_GRID_ENABLE_64_BIT_ATOMICS 1
#define HASH_GRID_USE_NORMALS 1

#include "SharcCommon.h"

namespace merian {

// One entry each in the hash-entry, accumulation and resolved buffers (40 bytes/voxel total).
struct Sharc {
    RWStructuredBuffer<HashGridKey> hash_entries;
    RWStructuredBuffer<SharcAccumulationData> accumulation;
    RWStructuredBuffer<SharcPackedData> resolved;

    // Per-frame uniforms.
    float3 camera_position;
    float3 prev_camera_position;
    float scene_scale;   // world-space voxel size control (larger => finer voxels)
    float radiance_scale; // quantization factor for the atomic u32 accumulation
    uint capacity;        // number of voxels (entries) in each buffer
    uint frame_index;
    uint accumulation_frame_num; // temporal accumulation window
    uint stale_frame_num_max;    // frames without new samples before eviction

    HashGridParameters grid_parameters() {
        HashGridParameters p;
        p.cameraPosition = camera_position;
        p.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
        p.sceneScale = scene_scale;
        p.levelBias = SHARC_GRID_LEVEL_BIAS;
        return p;
    }

    SharcParameters parameters() {
        SharcParameters p;
        p.hashGridParameters = grid_parameters();
        p.hashGridData.capacity = capacity;
        p.hashGridData.hashEntriesBuffer = hash_entries;
        p.radianceScale = radiance_scale;
        p.accumulationBuffer = accumulation;
        p.resolvedBuffer = resolved;
        return p;
    }

    // Resolved cached radiance at a hit, or 0 on a miss.
    float3 get(const float3 pos, const float3 normal) {
        SharcHitData hit;
        hit.positionWorld = pos;
        hit.normalWorld = normal;
        float3 radiance;
        if (SharcGetCachedRadiance(parameters(), hit, radiance, false))
            return radiance;
        return float3(0);
    }

    // Accumulate one radiance sample into the voxel (unit weight; resolve averages by sample count).
    void update(const float3 pos, const float3 normal, const float3 value) {
        SharcParameters params = parameters();
        HashGridKey key;
        HashGridIndex index =
            HashGridInsertEntry(params.hashGridData, pos, normal, params.hashGridParameters, key);
        SharcAddVoxelData(params, index, value, float3(1), float3(0, 0, 1), 0.0, 1);
    }

    // Compute-pass body: combine this frame's accumulation with the resolved history.
    void resolve(const uint entry_index) {
        SharcResolveParameters rp;
        rp.cameraPositionPrev = prev_camera_position;
        rp.accumulationFrameNum = accumulation_frame_num;
        rp.responsiveFrameNum = accumulation_frame_num;
        rp.staleFrameNumMax = stale_frame_num_max;
        rp.frameIndex = frame_index;
        SharcResolveEntry(entry_index, parameters(), rp);
    }

    // --- Debug ---
    float3 debug_hash_color(const float3 pos, const float3 normal) {
        return HashGridDebugColoredHash(pos, normal, grid_parameters());
    }

    uint voxel_level(const float3 pos) {
        return HashGridGetLevel(pos, grid_parameters());
    }
}

} // namespace merian
