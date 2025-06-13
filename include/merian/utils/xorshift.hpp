#pragma once

#include <cstdint>

namespace merian {

// Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
class XORShift32 {
  public:
    XORShift32(uint32_t seed) : state(seed) {}

    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    // returns a double
    float next_float() {
      return static_cast<float>(next() >> 8) * 5.9604644775390625e-08f;
    }

    // returns a double
    double next_double() {
        return static_cast<double>(next()) / 4294967296.0;
    }

  private:
    uint32_t state;
};

} // namespace merian
