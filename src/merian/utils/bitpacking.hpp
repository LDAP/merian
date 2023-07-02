#pragma once
#include <stdint.h>

namespace merian {

float half_to_float(uint16_t hi) noexcept;

uint16_t float_to_half_aprox(float fi) noexcept;

uint32_t pack_uint32(const uint16_t& lower, const uint16_t& upper) noexcept;

} // namespace merian
