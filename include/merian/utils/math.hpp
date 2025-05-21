#pragma once

#include <cstdint>
#include <numeric>
#include <vector>

namespace merian {

// Calculates the lowest common multiple of two numbers
inline uint32_t lcm(uint32_t a, uint32_t b) noexcept {
    return (a * b) / std::gcd(a, b);
}

// Calculates the lowest common multiple of numbers
inline uint32_t lcm(std::vector<uint32_t> numbers) noexcept {
    if (numbers.empty())
        return 0;
    if (numbers.size() == 1) {
        return numbers[0];
    }

    uint32_t cur = lcm(numbers[0], numbers[1]);

    for (uint32_t i = 2; i < numbers.size(); i++) {
        cur = lcm(cur, numbers[i]);
    }

    return cur;
}

inline uint32_t round_up(const uint32_t number, const uint32_t multiple) noexcept {
    return ((number + multiple / 2) / multiple) * multiple;
}

} // namespace merian
