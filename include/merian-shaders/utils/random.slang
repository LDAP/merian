#pragma once

import "merian-shaders/utils/bit-twiddling.slang";

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
        state ^= (state << uint(13));
        state ^= (state >> uint(17));
        state ^= (state << uint(5));

        return state;
    }
}

}
