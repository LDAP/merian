__exported import merian_shaders.utils.random;

namespace merian {

// Weighted reservoir sampling (Chao 1982) for RIS (Talbot et al. 2005)
struct StreamingResampler {
    float weight_sum;

    __init() {
        weight_sum = 0.f;
    }

    [mutating] bool update(const float weight, inout RandomGenerator rng) {
        if (!(weight > 0.f)) {
            return false;
        }
        weight_sum += weight;
        return rng.next_float() * weight_sum < weight;
    }
}

} // namespace merian
