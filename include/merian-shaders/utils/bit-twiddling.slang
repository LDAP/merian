namespace merian {

uint next_power_of_two(uint v) {
    v--;
    v |= (v >> uint(1));
    v |= (v >> uint(2));
    v |= (v >> uint(4));
    v |= (v >> uint(8));
    v |= (v >> uint(16));
    v++;
    return v;
}

// k counts from 0 at the lowest set bit and must be below countbits(bits).
uint select_set_bit(uint bits, uint k) {
    uint position = 0;
    for (uint width = 16; width > 0; width >>= 1) {
        const uint low = countbits(bits & ((1u << width) - 1u));
        if (k >= low) {
            k -= low;
            bits >>= width;
            position += width;
        }
    }
    return position;
}

// Converts a random uint to a random float in [0, 1).
float uint_to_zero_one_float_bits(uint source) {
    // 127 exponent, random mantissa => generate numbers in [1, 2), then subtract one
    return asfloat(1065353216u | (source >> uint(9))) - 1.0f;
}

// Converts a random uint to a random float in [0, 1).
// https://github.com/rust-random/rand/blob/7aa25d577e2df84a5156f824077bb7f6bdf28d97/src/distributions/float.rs#L111-L117
// Seems to retain a bit more bits compared to _bits, seems plausible since we shift with 8 here and with 9 above.
float uint_to_zero_one_float_rust(uint source) {
    // 32 - 8 = 24 (exponent), 1.0 / (1 << 24) = 5.9604644775390625e-08
    // top bits are usually more random.
    return float(source >> uint(8)) * 5.9604644775390625e-08f;
}

// Converts a random uint to a random float in [0, 1).
// Has issues: sometimes reaches 1
float uint_to_zero_one_float_naive(uint source) {
    return float(source) / 4294967296.0f;
}

}

