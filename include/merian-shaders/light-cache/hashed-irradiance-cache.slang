import merian_shaders.colors.colorspaces;
import merian_shaders.utils.hash_grid;
import merian_shaders.utils.random;

namespace merian {

static const uint16_t HASHED_IRR_CACHE_MAX_N = 256;
static const float HASHED_IRR_CACHE_MAX_VALUE = float16_t.maxValue;

struct HashedIrradianceCacheData {
    half3 irr;
    uint16_t N;
    // second moment of the accumulated luminance; full precision, half loses the tail
    float m2;
}

half3 hashed_irradiance_cache_sanitize(const half3 irr) {
    return select(any(isinf(irr)) || any(isnan(irr)), half3(0), irr);
}

struct HashedIrradianceCacheUpdate : HashGridUpdateFunction<HashedIrradianceCacheData> {
    float3 irr;

    void update(inout HashedIrradianceCacheData data) {
        if (any(isnan(irr))) {
            return;
        }

        const float3 sample = clamp(irr, float3(0), float3(HASHED_IRR_CACHE_MAX_VALUE));

        float3 mean = float3(data.irr);
        if (any(isinf(mean)) || any(isnan(mean))) {
            mean = sample;
            data.N = uint16_t(0);
            data.m2 = 0.0;
        }

        data.N = min(uint16_t(data.N + uint16_t(1)), HASHED_IRR_CACHE_MAX_N);
        const float rate = 1.0 / float(data.N);
        data.irr = half3(lerp(mean, sample, rate));
        const float l = yuv_luminance(sample);
        data.m2 = lerp(data.m2, l * l, rate);
    }
}

struct HashedIrradianceCache<let BUFFER_SIZE : uint,
                             let PROBE_COUNT : uint = 4,
                             let STOCHASTIC_INTERPOLATION : bool = false,
                             let SPLIT_STORAGE : bool = true,
                             let LOCALITY_BITS : uint = 0> {
    HashGrid<HashedIrradianceCacheData,
             BUFFER_SIZE,
             PROBE_COUNT,
             STOCHASTIC_INTERPOLATION,
             false,
             SPLIT_STORAGE,
             2,
             1,
             LOCALITY_BITS>
        grid;

    half3
    get(const float3 pos, const float3 normal, const float3 cam_pos, inout RandomGenerator rng) {
        if (let r = grid.get(pos, normal, cam_pos, rng)) {
            return hashed_irradiance_cache_sanitize(r.irr);
        }
        return half3(0);
    }

    // grid_pos is invalid unless the queried level matched.
    half3 get(const float3 pos,
              const float3 normal,
              const float3 cam_pos,
              const float3 prev_cam_pos,
              inout RandomGenerator rng,
              out HashGridPos grid_pos) {
        if (let r = grid.get(pos, normal, cam_pos, prev_cam_pos, rng, grid_pos)) {
            return hashed_irradiance_cache_sanitize(r.irr);
        }
        return half3(0);
    }

    // How many samples stand behind the value.
    half3 get(const float3 pos,
              const float3 normal,
              const float3 cam_pos,
              const float3 prev_cam_pos,
              inout RandomGenerator rng,
              out HashGridPos grid_pos,
              out uint samples) {
        samples = 0;
        if (let r = grid.get(pos, normal, cam_pos, prev_cam_pos, rng, grid_pos)) {
            samples = uint(r.N);
            return hashed_irradiance_cache_sanitize(r.irr);
        }
        return half3(0);
    }

    // Mean of the accumulated luminance and the squared standard error of that mean; a cell with
    // nothing behind it reports an unbounded error.
    bool estimate(const HashGridPos grid_pos, out float mean, out float se2) {
        mean = 0.0;
        se2 = 1e30;
        if (let r = grid.load(grid_pos)) {
            mean = yuv_luminance(float3(r.irr));
            const float n = float(r.N);
            if (n < 2.0) {
                return true;
            }
            se2 = max(r.m2 - mean * mean, 0.0) / n;
            return true;
        }
        return false;
    }

    // A valid grid_pos skips the probe, so it must locate the same cell.
    void update(const float3 pos,
                const float3 normal,
                const float3 irr,
                const float3 cam_pos,
                const float3 prev_cam_pos,
                const uint frame,
                inout RandomGenerator rng,
                const HashGridPos grid_pos = HashGridPos()) {
        HashedIrradianceCacheUpdate fn = {irr};
        if (grid_pos.is_valid()) {
            grid.update_at(grid_pos, frame, fn);
        } else {
            grid.update(pos, normal, cam_pos, prev_cam_pos, frame, fn, rng);
        }
    }
}

} // namespace merian
