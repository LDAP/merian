#pragma once

#include <bit>
#include <stdint.h>

namespace merian {

namespace internal {

typedef union FP32 {
    uint32_t u;
    float f;
} FP32;
typedef union FP16 {
    uint16_t u;
} FP16;

} // namespace internal

// Adapted from https://gist.github.com/rygorous/2156668#file-gistfile1-cpp-L574
// @Fabian "ryg" Giesen.
constexpr float half_to_float(const uint16_t hi) noexcept {
    internal::FP16 h = {hi};
    constexpr internal::FP32 magic = {113 << 23};
    constexpr uint32_t shifted_exp = 0x7c00 << 13; // exponent mask after shift
    internal::FP32 o;

    o.u = (h.u & 0x7fff) << 13;       // exponent/mantissa bits
    uint32_t exp = shifted_exp & o.u; // just the exponent
    o.u += (127 - 15) << 23;          // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp)      // Inf/NaN?
        o.u += (128 - 16) << 23; // extra exp adjust
    else if (exp == 0)           // Zero/Denormal?
    {
        o.u += 1 << 23; // extra exp adjust
        o.f -= magic.f; // renormalize
    }

    o.u |= (h.u & 0x8000) << 16; // sign bit
    return o.f;
}

// Adapted from https://gist.github.com/rygorous/2156668#file-gistfile1-cpp-L285
// @Fabian "ryg" Giesen.
constexpr uint16_t float_to_half(const float fi) noexcept {
    constexpr uint32_t f32infty = 255 << 23;
    constexpr uint32_t f16infty = 31 << 23;
    constexpr float magic = std::bit_cast<float, uint32_t>(15 << 23);
    constexpr uint32_t sign_mask = 0x80000000u;
    constexpr uint32_t no_sign_mask = ~(0x80000000u);
    constexpr uint32_t round_mask = ~0xfffu;

    internal::FP32 f = {.u = std::bit_cast<uint32_t>(fi) & no_sign_mask};
    const uint16_t sign = (std::bit_cast<uint32_t>(fi) & sign_mask) >> 16;

    // NOTE all the integer compares in this function can be safely
    // compiled into signed compares since all operands are below
    // 0x80000000. Important if you want fast straight SSE2 code
    // (since there's no unsigned PCMPGTD).
    if (f.u > f32infty) {
        return 0x7e00 | sign;
    } else if (f.u == f32infty)
        return 0x7c00 | sign;
    else [[likely]] {
        f.u &= round_mask;
        f.f *= magic;
        f.u -= round_mask;
        if (f.u > f16infty)
            f.u = f16infty; // Clamp to signed infinity if overflowed

        return (f.u >> 13) | sign;
    }
}

constexpr uint32_t pack_uint32(const uint16_t& lower, const uint16_t& upper) noexcept {
    return lower | (upper << 16);
}

} // namespace merian
