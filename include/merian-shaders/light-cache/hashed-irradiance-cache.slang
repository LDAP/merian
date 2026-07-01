import merian_shaders.utils.hash_grid;
import merian_shaders.utils.random;

namespace merian {

static const uint16_t HASHED_IRR_CACHE_MAX_N = 128;
static const float HASHED_IRR_CACHE_MIN_ALPHA = 0.01;

struct HashedIrradianceCacheData {
    half3 irr;
    uint16_t N;
}

struct HashedIrradianceCacheUpdate : HashGridUpdateFunction<HashedIrradianceCacheData> {
    half3 irr;

    void update(inout HashedIrradianceCacheData data) {
        if (any(isinf(data.irr)) || any(isnan(data.irr))) {
            data.N = 1;
            data.irr = irr;
        }

        data.N = min(uint16_t(data.N + uint16_t(1)), HASHED_IRR_CACHE_MAX_N);
        data.irr = half3(
            lerp(float3(data.irr), float3(irr), max(1.0 / float(data.N), HASHED_IRR_CACHE_MIN_ALPHA)));
    }
}

struct HashedIrradianceCache<let BUFFER_SIZE : uint,
                             let PROBE_COUNT : uint = 4,
                             let STOCHASTIC_INTERPOLATION : bool = false,
                             let NORMAL_BITS : uint = 0> {
    HashGrid<HashedIrradianceCacheData, BUFFER_SIZE, PROBE_COUNT, STOCHASTIC_INTERPOLATION,
             NORMAL_BITS>
        grid;

    half3
    get(const float3 pos, const float3 normal, const float3 cam_pos, inout RandomGenerator rng) {
        if (let r = grid.get(pos, normal, cam_pos, rng)) {
            return r.irr;
        }
        return half3(0);
    }

    half3 get(const float3 pos,
              const float3 normal,
              const float3 cam_pos,
              const float3 prev_cam_pos,
              inout RandomGenerator rng) {
        if (let r = grid.get(pos, normal, cam_pos, prev_cam_pos, rng)) {
            return r.irr;
        }
        return half3(0);
    }

    void update(const float3 pos,
                const float3 normal,
                const float3 irr,
                const float3 cam_pos,
                const float3 prev_cam_pos,
                const uint frame,
                inout RandomGenerator rng) {
        HashedIrradianceCacheUpdate fn = {irr};
        grid.update(pos, normal, cam_pos, prev_cam_pos, frame, fn, rng);
    }
}

} // namespace merian
