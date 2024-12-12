#ifndef _MERIAN_SHADERS_COMMON_H_
#define _MERIAN_SHADERS_COMMON_H_

#define M_PI   3.14159265358979323846
#define TWO_PI 6.283185307179586
#define FOUR_PI 12.566370614359172
#define INV_PI 0.3183098861837907
#define INV_TWO_PI 0.15915494309189535
#define INV_FOUR_PI 0.07957747154594767
#define INV_SQRT_TWO_PI 0.3989422804014327

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

#endif // _MERIAN_SHADERS_COMMON_H_
