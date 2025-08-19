#pragma once

// #define MERIAN_WORKGROUP_INDEX                                                                     \
//     (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y)
//
// #define MERIAN_GLOBAL_INVOCATION_INDEX \
//     (MERIAN_WORKGROUP_INDEX * gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z +
//     gl_LocalInvocationIndex)

namespace merian {

float square(const float x) {
    return x * x;
}

// returns 1/x if x > 0 else 1.
float safe_reciprocal(const float x) {
    return (x > 0. ? 1. / x : 1.0);
}

bool relative_distance_greather_than(float3 center, float3 p1, float3 p2, float threshold) {
    float d1 = distance(center, p1);
    float d2 = distance(center, p2);
    return (abs(d1 - d2) / max(d1, d2)) > threshold;
}

// log(x + 1); David Goldberg (1991). What every computer scientist should know about floating-point
// arithmetic.
float log1p(float x) {
    float u = x + 1.0f;
    if (u == 1.0f) {
        return x;
    }
    float y = log(u);
    if (x < 1.0f) {
        return (x * y) / (u - 1.0f);
    }
    return y;
}

// exp(x) - 1; Nicholas J. Higham (2002). Accuracy and Stability of Numerical Algorithms. Society
// for Industrial and Applied Mathematics
float expm1(float x) {
    float u = exp(x);
    if (u == 1.0f) {
        return x;
    }
    float y = u - 1.0f;
    if (abs(x) < 1.0f) {
        return (x * y) / log(u);
    }
    return y;
}

// x / (exp(x) - 1); Nicholas J. Higham (2002). Accuracy and Stability of Numerical Algorithms.
// Society for Industrial and Applied Mathematics
float x_over_expm1(float x) {
    float u = exp(x);
    if (u == 1.0f) {
        return 1.0f;
    }
    float y = u - 1.0f;
    if (abs(x) < 1.0f) {
        return log(u) / y;
    }
    return x / y;
}

// retains the sign when computing pow()
float signpow(float a, float x) {
    return pow(abs(a), x) * sign(a);
}

// cube root
float cbrt(float a) {
    return signpow(a, 1. / 3);
}

}
