#pragma once

#include <cstdint>

namespace merian {

// Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs"
class XORShift32 {
  public:
    XORShift32(uint32_t seed) : state(seed) {}

    // returns a double
    double get() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state / 4294967296.0;
    }

  private:
    uint32_t state;
};

} // namespace merian
