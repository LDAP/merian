#ifndef _MERIAN_SHADERS_COLORS_REC2020_H_
#define _MERIAN_SHADERS_COLORS_REC2020_H_


vec3 rec2020_to_xyY(vec3 rec2020) {
  const mat3 rec2020_to_xyz = mat3(
    6.36958048e-01, 2.62700212e-01, 4.20575872e-11,
    1.44616904e-01, 6.77998072e-01, 2.80726931e-02,
    1.68880975e-01, 5.93017165e-02, 1.06098506e+00);
  const vec3 xyz = rec2020_to_xyz * rec2020;
  return vec3(xyz.xy / dot(vec3(1),xyz), xyz.y);
}

vec3 xyY_to_rec2020(vec3 xyY) {
  const vec3 xyz = vec3(xyY.xy, 1.0-xyY.x-xyY.y) * xyY.z / xyY.y;
  const mat3 xyz_to_rec2020 = mat3(
    1.71665119, -0.66668435,  0.01763986,
   -0.35567078,  1.61648124, -0.04277061,
   -0.25336628,  0.01576855,  0.94210312);
  return xyz_to_rec2020 * xyz;
}

vec3 rec709_to_rec2020(const vec3 rec709) {
  const mat3 M_rec709_to_rec2020 = mat3(
    0.62750375, 0.06910828, 0.01639406,
    0.32927542, 0.91951916, 0.08801125,
    0.04330266, 0.0113596 , 0.89538035);
  return M_rec709_to_rec2020 * rec709;
}

vec3 rec2020_to_rec709(const vec3 rec2020) {
  const mat3 M_rec2020_to_rec709 = mat3(
    1.66023  ,  -0.124553  ,  -0.0181551,
   -0.587548 ,   1.13293   ,  -0.100603,
   -0.0728382,  -0.00834963,   1.119);
  return M_rec2020_to_rec709 * rec2020;
}

#endif
