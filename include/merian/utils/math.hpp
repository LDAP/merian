#pragma once

#include "merian/utils/vector_matrix.hpp"

#include <cstdint>
#include <numeric>
#include <span>
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

inline uint32_t wrap(const int32_t value, const uint32_t size) noexcept {
    const auto n = static_cast<int32_t>(size);
    return static_cast<uint32_t>(((value % n) + n) % n);
}

inline float triangle_area(const float3& a, const float3& b, const float3& c) noexcept {
    return 0.5f * length(cross(b - a, c - a));
}

inline bool convex_polygon_contains(const std::span<const float2> polygon,
                                    const float2& point) noexcept {
    float winding = 0;
    for (size_t i = 0; i < polygon.size(); i++) {
        const float2 edge = polygon[(i + 1) % polygon.size()] - polygon[i];
        const float2 to_point = point - polygon[i];
        const float cross = edge.x * to_point.y - edge.y * to_point.x;
        if (cross == 0)
            continue;
        if (winding == 0)
            winding = cross;
        else if ((cross > 0) != (winding > 0))
            return false;
    }
    return true;
}

} // namespace merian
