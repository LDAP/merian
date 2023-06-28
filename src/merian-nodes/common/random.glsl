#ifndef _RANDOM_H_
#define _RANDOM_H_

// Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
// Returns random float in [0, 1) and updates state.
float XorShift32(inout uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state / 4294967296.0;
}

vec4 XorShift32Vec4(inout uint state) {
    return vec4(XorShift32(state), XorShift32(state), XorShift32(state), XorShift32(state));
}

#endif
