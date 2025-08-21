
#extension GL_GOOGLE_include_directive : require


#ifndef _MERIAN_SHADERS_RAYDIFFERENTIAL_H_
#define _MERIAN_SHADERS_RAYDIFFERENTIAL_H_

struct RayDifferential {
    // current differential of origin
    vec3 dOdx;
    vec3 dOdy;
    // current differential of direction
    vec3 dDdx;
    vec3 dDdy;
};

RayDifferential ray_diff_create(const vec3 dOdx, const vec3 dOdy, const vec3 dDdx, const vec3 dDdy) {
    const RayDifferential rd = {dOdx, dOdy, dDdx, dDdy};
    return rd;
}


void ray_diff_propagate(inout RayDifferential rd,
                        const vec3 ray_dir,
                        const float t,
                        const vec3 hit_normal) {
    rd.dOdx += t * rd.dDdx;  // Part of Igehy Equation 10.
    rd.dOdy += t * rd.dDdy;

    const float rcpDN = 1.0f / dot(ray_dir, hit_normal);       // Igehy Equations 10 and 12.
    rd.dOdx += ray_dir * -dot(rd.dOdx, hit_normal) * rcpDN;
    rd.dOdy += ray_dir * -dot(rd.dOdy, hit_normal) * rcpDN;
}


#endif // _MERIAN_SHADERS_RAYDIFFERENTIAL_H_
