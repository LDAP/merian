#pragma once

#include "glm/glm.hpp"
namespace merian {

// Variation of smoothstep from Ken Perlin
inline double smootherstep(double x) {
    return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
}

template<typename T>
inline T evaluate_bezier(float t, const T& p0, const T& p1, const T& p2) {
    const float u = 1.f - t;
    const float tt = t * t;
    const float uu = u * u;

    T p{0};

    p = uu * p0; // first term
    p += 2 * u * t * p1;       // second term
    p += tt * p2;              // third term

    return p;
}

} // namespace merian
