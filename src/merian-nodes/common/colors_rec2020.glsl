vec3 rec2020_to_xyY(vec3 rec2020)
{
  const mat3 rec2020_to_xyz = mat3(
    6.36958048e-01, 2.62700212e-01, 4.20575872e-11,
    1.44616904e-01, 6.77998072e-01, 2.80726931e-02,
    1.68880975e-01, 5.93017165e-02, 1.06098506e+00);
  const vec3 xyz = rec2020_to_xyz * rec2020;
  return vec3(xyz.xy / dot(vec3(1),xyz), xyz.y);
}

vec3 xyY_to_rec2020(vec3 xyY)
{
  const vec3 xyz = vec3(xyY.xy, 1.0-xyY.x-xyY.y) * xyY.z / xyY.y;
  const mat3 xyz_to_rec2020 = mat3(
    1.71665119, -0.66668435,  0.01763986,
   -0.35567078,  1.61648124, -0.04277061,
   -0.25336628,  0.01576855,  0.94210312);
  return xyz_to_rec2020 * xyz;
}
