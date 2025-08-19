
#ifndef _MERIAN_SHADERS_COMMON_H_
#define _MERIAN_SHADERS_COMMON_H_

#define M_PI   3.14159265358979323846
#define TWO_PI 6.283185307179586
#define FOUR_PI 12.566370614359172
#define INV_PI 0.3183098861837907
#define INV_TWO_PI 0.15915494309189535
#define INV_FOUR_PI 0.07957747154594767
#define INV_SQRT_TWO_PI 0.3989422804014327

#define FLT_EPSILON 1.192092896e-07F

// returns 1/x if x > 0 else 1.
#define MERIAN_SAFE_RECIPROCAL(x) (x > 0. ? 1. / x : 1.0)

#define MERIAN_SQUARE(x) ((x) * (x))

float merian_square(const float x) {
    return x * x;
}

#define MERIAN_WORKGROUP_INDEX (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y)
#define MERIAN_GLOBAL_INVOCATION_INDEX (MERIAN_WORKGROUP_INDEX * gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z + gl_LocalInvocationIndex)

bool merian_relative_distance_greather_than(const vec3 center, const vec3 p1, const vec3 p2, const float threshold) {
    const float d1 = distance(center, p1);
    const float d2 = distance(center, p2);
    return abs(d1 - d2) / max(d1, d2) > threshold;
}

// log(x + 1); David Goldberg (1991). What every computer scientist should know about floating-point arithmetic.
float log1p (const float x) {
    volatile const float u = x + 1.0f;
    if (u == 1.0f) {
        return x;
    }
    const float y = log(u);
    if (x < 1.0f) {
        return x * y / (u - 1.0f);
    }
    return y;
}

// exp(x) - 1; Nicholas J. Higham (2002). Accuracy and Stability of Numerical Algorithms. Society for Industrial and Applied Mathematics
float expm1(const float x) {
    const float u = exp(x);
    if (u == 1.0f) {
        return x;
    }
    const float y = u - 1.0f;
    if (abs(x) < 1.0f) {
        return x * y / log(u);
    }
    return y;
}

// x / (exp(x) - 1); Nicholas J. Higham (2002). Accuracy and Stability of Numerical Algorithms. Society for Industrial and Applied Mathematics
float x_over_expm1(const float x) {
    const float u = exp(x);
    if (u == 1.0f) {
        return 1.0f;
    }
    const float y = u - 1.0f;
    if (abs(x) < 1.0f) {
        return log(u) / y;
    }
    return x / y;
}

// retains the sign when computing pow()
float signpow(const float a, const float x) {
    return pow(abs(a), x) * sign(a);
}

// cube root
float cbrt(const float a) {
    return signpow(a, 0.333333333333);
}

#endif // _MERIAN_SHADERS_COMMON_H_
