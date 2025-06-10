#ifndef _MERIAN_SHADERS_RANDOM_H_
#define _MERIAN_SHADERS_RANDOM_H_

// Converts a random uint to a random float in [0, 1).
float uint_to_zero_one_float(const uint source) {
    // 127 exponent, random mantissa => generate numbers in [1, 2), then subtract one
    return uintBitsToFloat(0x3f800000u | (source >> 9)) - 1.0;
}

// Converts a random uint to a random float in [0, 1).
// https://github.com/rust-random/rand/blob/7aa25d577e2df84a5156f824077bb7f6bdf28d97/src/distributions/float.rs#L111-L117
float uint_to_zero_one_float_rust(const uint source) {
    // 32 - 8 = 24 (exponent), 1.0 / (1 << 24) = 5.9604644775390625e-08
    // top bits are usually more random.
    return (source >> 8) * 5.9604644775390625e-08;
}

// Converts a random uint to a random float in [0, 1).
float uint_to_zero_one_float_naive(const uint source) {
    return source / 4294967296.0;
}

// Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
// Returns random float in [0, 1) and updates state.

float XorShift32(inout uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return uint_to_zero_one_float(state);
}

#define XorShift32Vec2(state) vec2(XorShift32(state), XorShift32(state))
#define XorShift32Vec3(state) vec3(XorShift32(state), XorShift32(state), XorShift32(state))
#define XorShift32Vec4(state) vec4(XorShift32(state), XorShift32(state), XorShift32(state), XorShift32(state))

// -------------------------------------------------------

#endif
