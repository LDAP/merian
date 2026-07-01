namespace merian {

// Van der Corput radical inverse of n in the given base, in [0, 1).
float radical_inverse(uint n, const uint base) {
    const float inv_base = 1.0f / base;
    float inv_bn = inv_base;
    float result = 0.0f;
    while (n > 0u) {
        result += float(n % base) * inv_bn;
        n /= base;
        inv_bn *= inv_base;
    }
    return result;
}

// Halton sequence point n in bases (2, 3), in [0, 1)^2.
float2 halton(const uint n) {
    return float2(radical_inverse(n, 2u), radical_inverse(n, 3u));
}

// First 8 points of the base-(2, 3) Halton sequence, in [0, 1)^2.
static const float2 HALTON8[8] = {
    float2(1.0f / 2.0f, 1.0f / 3.0f), float2(1.0f / 4.0f, 2.0f / 3.0f),
    float2(3.0f / 4.0f, 1.0f / 9.0f), float2(1.0f / 8.0f, 4.0f / 9.0f),
    float2(5.0f / 8.0f, 7.0f / 9.0f), float2(3.0f / 8.0f, 2.0f / 9.0f),
    float2(7.0f / 8.0f, 5.0f / 9.0f), float2(1.0f / 16.0f, 8.0f / 9.0f),
};

// Precomputed base-(2, 3) Halton point, wrapping every 8 samples.
float2 halton8(const uint n) {
    return HALTON8[n % 8u];
}

// R2 low-discrepancy sequence (Roberts 2018): point n of the plastic-constant lattice, in [0, 1)^2.
float2 r2_sequence(const uint n) {
    const float2 alpha = float2(0.7548776662466927f, 0.5698402909980532f); // (1 / g, 1 / g^2)
    return frac(0.5f + alpha * float(n));
}

} // namespace merian
