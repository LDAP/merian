#pragma once
#include <stdint.h>

namespace merian {

float half_to_float(uint16_t hi);

uint16_t float_to_half(float fi);

} // namespace merian
