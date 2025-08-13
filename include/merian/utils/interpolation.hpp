#pragma once

#include <concepts>

namespace merian {

// Variation of smoothstep from Ken Perlin
template <typename T>
    requires std::floating_point<T>
inline T smootherstep(T x) {
    return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
}

template <typename P>
inline P evaluate_bezier(const float t, const P& p0, const P& p1, const P& p2) {
    const float u = 1.f - t;
    const float tt = t * t;
    const float uu = u * u;

    P p{0};

    p = uu * p0;         // first term
    p += 2 * u * t * p1; // second term
    p += tt * p2;        // third term

    return p;
}

} // namespace merian
