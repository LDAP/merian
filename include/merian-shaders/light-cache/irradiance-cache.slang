__exported import merian_shaders.utils.random;

namespace merian {

interface IrradianceCache {
    half3 get(const float3 pos,
              const float3 normal,
              const float3 cam_pos,
              const float3 prev_cam_pos,
              inout RandomGenerator rng);
}

struct NullIrradianceCache : IrradianceCache {
    half3 get(const float3 pos,
              const float3 normal,
              const float3 cam_pos,
              const float3 prev_cam_pos,
              inout RandomGenerator rng) {
        return half3(0);
    }
}

} // namespace merian
