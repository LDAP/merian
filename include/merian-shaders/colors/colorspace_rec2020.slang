#pragma once

implementing colorspaces;

namespace merian {

public float3 rec2020_to_xyY(float3 rec2020) {
    float3 xyz = mul(rec2020, float3x3(float3(6.36958048e-01, 2.62700212e-01, 4.20575872e-11),
                                       float3(1.44616904e-01, 6.77998072e-01, 2.80726931e-02),
                                       float3(1.68880975e-01, 5.93017165e-02, 1.06098506e+00)));
    return float3(xyz.xy / dot(float3(1.f), xyz), xyz.y);
}

public float3 xyY_to_rec2020(float3 xyY) {
    float3 xyz = (float3(xyY.xy, (1.0f - xyY.x) - xyY.y) * xyY.z) / xyY.y.xxx;
    return mul(xyz, float3x3(float3(1.71665119, -0.66668435, 0.01763986),
                             float3(-0.35567078, 1.61648124, -0.04277061),
                             float3(-0.25336628, 0.01576855, 0.94210312)));
}

public float3 rec709_to_rec2020(float3 rec709) {
    return mul(rec709, float3x3(float3(0.62750375, 0.06910828, 0.01639406),
                                float3(0.32927542, 0.91951916, 0.08801125),
                                float3(0.04330266, 0.0113596, 0.89538035)));
}

public float3 rec2020_to_rec709(float3 rec2020) {
    return mul(rec2020, float3x3(float3(1.66023, -0.124553, -0.0181551),
                                 float3(-0.587548, 1.13293, -0.100603),
                                 float3(-0.0728382, -0.00834963, 1.119)));
}

}
