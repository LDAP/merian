#pragma once

namespace merian {

// Returns a matrix of axis (tangent, bitangent, z) which all being perpendicular to each other
// Follows Building an Orthonormal Basis, Revisited, Duff et al. 2017.
// 
// The basis vectors are in the rows!
float3x3 make_frame(const float3 z) {
    const float _sign = (z.z >= 0) ? 1 : -1;
    const float a = -1.0 / (_sign + z.z);
    const float b = z.x * z.y * a;
    return float3x3(float3(1.0 + _sign * z.x * z.x * a, _sign * b, -_sign * z.x),
                    float3(b, _sign + z.y * z.y * a, -z.y), z);
}

// Returns a matrix of axis (tangent/x, bitangent/y, z) which all being perpendicular to each other
// Follows Building an Orthonormal Basis, Revisited, Duff et al. 2017
void make_frame(const float3 z, out float3 x, out float3 y) {
    const float _sign = (z.z >= 0) ? 1 : -1;
    const float a = -1.0 / (_sign + z.z);
    const float b = z.x * z.y * a;
    x = float3(1.0 + _sign * z.x * z.x * a, _sign * b, -_sign * z.x);
    y = float3(b, _sign + z.y * z.y * a, -z.y);
}

// Returns a matrix of axis (x, y, z) which all being perpendicular to each other
//
// The basis vectors are in the rows!
float3x3 make_frame_naive(const float3 z) {
    const float3 up = abs(z.x) > abs(z.y) ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 du = normalize(cross(up, z));
    return float3x3(float3(du), float3(normalize(cross(du, z))), float3(z));
}

// needs orthonormal frame
float3 world_to_frame(const float3x3 frame, const float3 v) {
    return float3(dot(frame[0], v), dot(frame[1], v), dot(frame[2], v));
}

float3 frame_to_world(const float3x3 frame, const float3 v) {
    return mul(v, frame);
}

struct Frame {
    // a orthonormal basis
    // row 0: normalized shading tangent
    // row 1: normalized shading bitangent
    // row 2: normalized shading normal
    float3x3 basis; 

    __init(const float3 n) {
        basis = make_frame(n);
    }

    float3 to_frame(const float3 v) {
        return world_to_frame(basis, v);
    }

    float3 to_world(const float3 v) {
        return frame_to_world(basis, v);
    }
}

}
