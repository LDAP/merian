#pragma once

namespace merian {

struct RayDifferential {
    // current differential of origin
    float3 dOdx;
    float3 dOdy;
    // current differential of direction
    float3 dDdx;
    float3 dDdy;

    [mutating]
    void ray_diff_propagate(const float3 ray_dir, const float t, const float3 hit_normal) {
        dOdx += t * dDdx; // Part of Igehy Equation 10.
        dOdy += t * dDdy;

        const float rcpDN = 1.0f / dot(ray_dir, hit_normal); // Igehy Equations 10 and 12.
        dOdx += ray_dir * -dot(dOdx, hit_normal) * rcpDN;
        dOdy += ray_dir * -dot(dOdy, hit_normal) * rcpDN;
    }
};

}
