#pragma once

// #define MERIAN_WORKGROUP_INDEX                                                                     \
//     (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y)
//
// #define MERIAN_GLOBAL_INVOCATION_INDEX \
//     (MERIAN_WORKGROUP_INDEX * gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z +
//     gl_LocalInvocationIndex)

namespace merian {

T pow2<T : IFloat>(const T x) {
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

/**
 * @brief      Calculate the reflection direction for an outgoing vector. Equal to reflect(-i, n).
 * All directions point away from the surface point.
 *
 * @param      i the outgoing vector
 * @param      n the surface normal
 *
 * @return     the reflected vector.
 */
float3 reflect2(const float3 i, const float3 n) {
    return 2.0f * dot(n, i) * n - i;
}

/**
 * @brief      Calculate the reflection direction for an outgoing vector. Equal to reflect(-i, n).
 * All directions point away from the surface point.
 *
 * @param      i the outgoing vector
 * @param      n the surface normal
 * @param      ndoti precomputed dot product dot(n, i)
 *
 * @return     the reflected vector.
 */
float3 reflect2(const float3 i, const float3 n, const float ndoti) {
    return 2.0f * ndoti * n - i;
}

float2x2 inverse(float2x2 m) {
    float2x2 inv;
    float invdet = 1.0f / determinant(m);

    inv[0][0] = m[1][1] * invdet;
    inv[1][1] = m[0][0] * invdet;
    inv[0][1] = -m[0][1] * invdet;
    inv[1][0] = -m[1][0] * invdet;

    return inv;
}

float3x3 inverse(float3x3 m) {
    float3x3 inv;
    float invdet = 1.0f / determinant(m);

    inv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
    inv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
    inv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet;
    inv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet;
    inv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet;

    return inv;
}

float4x4 inverse(float4x4 m) {
    float n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    float n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    float n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    float n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    float t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 -
                n23 * n32 * n44 + n22 * n33 * n44;
    float t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 +
                n13 * n32 * n44 - n12 * n33 * n44;
    float t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 -
                n13 * n22 * n44 + n12 * n23 * n44;
    float t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 +
                n13 * n22 * n34 - n12 * n23 * n34;

    float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    float idet = 1.0f / det;

    float4x4 ret;

    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 +
                 n23 * n31 * n44 - n21 * n33 * n44) *
                idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 -
                 n22 * n31 * n44 + n21 * n32 * n44) *
                idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 +
                 n22 * n31 * n43 - n21 * n32 * n43) *
                idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 -
                 n13 * n31 * n44 + n11 * n33 * n44) *
                idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 +
                 n12 * n31 * n44 - n11 * n32 * n44) *
                idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 -
                 n12 * n31 * n43 + n11 * n32 * n43) *
                idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 +
                 n13 * n21 * n44 - n11 * n23 * n44) *
                idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 -
                 n12 * n21 * n44 + n11 * n22 * n44) *
                idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 +
                 n12 * n21 * n43 - n11 * n22 * n43) *
                idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 -
                 n13 * n21 * n34 + n11 * n23 * n34) *
                idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 +
                 n12 * n21 * n34 - n11 * n22 * n34) *
                idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 -
                 n12 * n21 * n33 + n11 * n22 * n33) *
                idet;

    return ret;
}

// Returns the Moore-Penrose inverse also known as the pseudoinverse A+ = (A^TA)^-1 * A^T.
// A+ * b is the solution that minimizes || Ax - b ||.
//
// For A = (v(2) - v(0), v(1) - v(0)) this is also the worldspace derivative of the barycentric
// coordinates. n = cross(A[0], A[1])
float3x2 pseudoinverse(const float2x3 A, float3 n) {
    // const mat3x2 A_T = transpose(A);
    // return inverse(A_T * A) * A_T;

    // this is a more efficient solution of the above:
    n /= dot(n, n);
    return transpose(float2x3(float3(cross(A[1], n)), float3(cross(n, A[0]))));
}

float3x2 pseudoinverse(const float2x3 A) {
    return pseudoinverse(A, cross(A[0], A[1]));
}

void pseudoinverse(float3 du, float3 dv, float3 cross_dudv, out float3 du_dx, out float3 dv_dx) {
    float3 nt = cross_dudv / dot(cross_dudv, cross_dudv).xxx;
    du_dx = cross(dv, nt);
    dv_dx = cross(nt, du);
}

// du = v(2) - v(0), dv = v(1) - v(0)
void pseudoinverse(float3 du, float3 dv, out float3 du_dx, out float3 dv_dx) {
    float3 param;
    float3 param_1;
    pseudoinverse(du, dv, cross(du, dv), param, param_1);
    du_dx = param;
    dv_dx = param_1;
}

}
