import "merian-shaders/utils/bit-twiddling.slang";
import merian_shaders.utils.hash;

namespace merian {

interface RandomGenerator {

    /**
     * Returns the next uniformly distributed pseudorandom sample.
     */
    [mutating]
    uint next();

    /**
     * Returns a uniformly distributed sample in [0, 1).
     */
    [mutating]
    float next_float() {
        return uint_to_zero_one_float_rust(next());
    }

    /**
     * Returns a uniformly distributed sample in [0, 1).
     */
    [mutating]
    float2 next_float2() {
        float2 sample;
        sample.x = uint_to_zero_one_float_rust(next());
        sample.y = uint_to_zero_one_float_rust(next());
        return sample;
    }

    /**
     * Returns a uniformly distributed sample in [0, 1).
     */
    [mutating]
    float3 next_float3() {
        float3 sample;
        sample.x = uint_to_zero_one_float_rust(next());
        sample.y = uint_to_zero_one_float_rust(next());
        sample.z = uint_to_zero_one_float_rust(next());
        return sample;
    }

    /**
     * Returns a uniformly distributed sample in [0, 1).
     */
    [mutating]
    float4 next_float4() {
        float4 sample;
        sample.x = uint_to_zero_one_float_rust(next());
        sample.y = uint_to_zero_one_float_rust(next());
        sample.z = uint_to_zero_one_float_rust(next());
        sample.w = uint_to_zero_one_float_rust(next());
        return sample;
    }

}

// Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
struct XorShift32 : RandomGenerator {
    uint state;

    __init(uint seed) {
        this.state = seed;
    }

    [mutating]
    uint next() {
        state = xorshift32(state);
        return state;
    }
}

struct Wang : RandomGenerator {
    uint state;

    __init(uint seed) {
        this.state = seed;
    }

    [mutating]
    uint next() {
        state = (state ^ 61) ^ (state >> 16);
        state *= 9;
        state = state ^ (state >> 4);
        state *= 0x27d4eb2d;
        state = state ^ (state >> 15);
        return state;
    }
}

struct PCG : RandomGenerator {
    uint state;

    __init(uint seed) {
        this.state = seed;
    }

    [mutating]
    uint next() {
        uint s = state;
        state = state * 747796405u + 2891336453u;
        uint word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
        return (word >> 22u) ^ word;
    }
}

}
