#ifndef _MERIAN_SHADERS_FRAMES_H_
#define _MERIAN_SHADERS_FRAMES_H_

// Returns a matrix of axis (tangent, bitangent, z) which all being perpendicular to each other
// Follows Building an Orthonormal Basis, Revisited, Duff et al. 2017
mat3x3 make_frame(const vec3 z) {
    const float sign = (z.z >= 0) ? 1 : -1;
    const float a = -1.0 / (sign + z.z);
    const float b = z.x * z.y * a;
    return mat3(vec3(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x),
                vec3(b, sign + z.y * z.y * a, -z.y),
                z);
}

void make_frame(const vec3 z, out vec3 x, out vec3 y) {
    const float sign = (z.z >= 0) ? 1 : -1;
    const float a = -1.0 / (sign + z.z);
    const float b = z.x * z.y * a;
    x = vec3(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
    y = vec3(b, sign + z.y * z.y * a, -z.y);
}

// Returns a matrix of axis (x, y, z) which all being perpendicular to each other
mat3x3 make_frame_naive(const vec3 z) {
    const vec3 up = (abs(z.x) > abs(z.y)) ? vec3(0,1,0) : vec3(1,0,0);
    const vec3 du = normalize(cross(up, z));
    return mat3(du, normalize(cross(du, z)), z);
}

// needs orthonormal frame
vec3 world_to_frame(const mat3x3 frame, const vec3 v) {
    return vec3(dot(frame[0], v), dot(frame[1], v), dot(frame[2], v));
}

vec3 frame_to_world(const mat3x3 frame, const vec3 v) {
    return frame * v;
}

#endif
