
// Computes a weight for the quality of the reprojection.
// Reprojections with normals further away than normal_reject_cos = cos(alpha)
// and depths farther apart that z_reject_percent * max(curr_z, prev_z) are completely rejected
float reprojection_weight(const vec3 curr_normal, const vec3 prev_normal, const float normal_reject_cos,
                          const float curr_z, const float prev_z, const float z_reject_percent) {

    return smoothstep(normal_reject_cos, 1.0, dot(curr_normal, prev_normal))                        // Similar normals
           * (1.0 - smoothstep(0.0, z_reject_percent * max(curr_z, prev_z), abs(curr_z - prev_z))); // Similar depth
}

// Like reprojection_weight but binary
bool reprojection_valid(const vec3 curr_normal, const vec3 prev_normal, const float normal_reject_cos,
                          const float curr_z, const float prev_z, const float z_reject_percent) {

    return normal_reject_cos <= dot(curr_normal, prev_normal)                 // Similar normals
           && abs(curr_z - prev_z) <= z_reject_percent * max(curr_z, prev_z); // Similar depth
}
