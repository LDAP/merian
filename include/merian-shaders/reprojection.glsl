#version 460

#extension GL_GOOGLE_include_directive : require


#ifndef _MERIAN_SHADERS_REPROJECTION
#define _MERIAN_SHADERS_REPROJECTION

// Computes a weight for the quality of the reprojection.
// Reprojections with normals further away than normal_reject_cos = cos(alpha)
// and depths farther apart that z_reject_percent * max(curr_z, prev_z) are completely rejected
float reprojection_weight(const vec3 curr_normal, const vec3 prev_normal, const float normal_reject_cos,
                          const float curr_z, const float vel_z, const float prev_z, const float z_reject_percent) {

    return smoothstep(normal_reject_cos, 1.0, dot(curr_normal, prev_normal))                        // Similar normals
                                                                             // move vel_z outside of abs to reject less
                      * (1.0 - smoothstep(0.0, z_reject_percent * max(curr_z, prev_z), abs(curr_z - prev_z + vel_z))); // Similar depth
}

float reprojection_weight(const vec3 curr_normal, const vec3 prev_normal, const float normal_reject_cos,
                          const float curr_z, const float vel_z, const ivec2 pixel_offset, const vec2 grad_z, 
                          const float prev_z, const float z_reject_percent) {

    return smoothstep(normal_reject_cos, 1.0, dot(curr_normal, prev_normal))                        // Similar normals
                      * (1.0 - smoothstep(0.0, z_reject_percent * max(curr_z, prev_z), abs(curr_z + dot(pixel_offset, grad_z) - prev_z + vel_z))); // Similar depth
}

// Like reprojection_weight but binary
bool reprojection_valid(const vec3 curr_normal, const vec3 prev_normal, const float normal_reject_cos,
                          const float curr_z,  const float vel_z, const ivec2 pixel_offset, const vec2 grad_z, const float prev_z, const float z_reject_percent) {

    return normal_reject_cos <= dot(curr_normal, prev_normal)                 // Similar normals
           && abs(curr_z + dot(pixel_offset, grad_z) - prev_z + vel_z) <= z_reject_percent * max(curr_z, prev_z); // Similar depth
}

bool reprojection_valid(const vec3 curr_normal, const vec3 prev_normal, const float normal_reject_cos,
                          const float curr_z,  const float vel_z, const float prev_z, const float z_reject_percent) {

    return normal_reject_cos <= dot(curr_normal, prev_normal)                 // Similar normals
           && abs(curr_z - prev_z + vel_z) <= z_reject_percent * max(curr_z, prev_z); // Similar depth
}

// Intersects the motion vector with the image borders.
// 
// Returns true if prev pos was outside the image borders.
// prev_pos = pos + mv
bool reprojection_intersect_border(inout vec2 prev_pos, const vec2 mv, const vec2 image_size_minus_one) {
    if (any(greaterThan(round(prev_pos), image_size_minus_one)) || any(lessThan(round(prev_pos), vec2(0)))) {
        // Intersect motion vector with image:
        const float tmin = max(min(prev_pos.x / mv.x, (prev_pos.x - image_size_minus_one.x) / mv.x),
                               min(prev_pos.y / mv.y, (prev_pos.y - image_size_minus_one.y) / mv.y));
        prev_pos = prev_pos - mv * tmin;
        return true;
    }
    return false;
}

// pixel = current_pixel + mv
ivec2 reproject_pixel_nearest(const vec2 pixel) {
    return ivec2(round(pixel));
}

// Performs stochastic bilinear interpolation when reprojecting
// pixel = current_pixel + mv
ivec2 reproject_pixel_stochastic(const vec2 pixel, const float random) {
    const vec2 relative_pos = fract(pixel);

    // (0, 0)
    float bary_sum = relative_pos.x * relative_pos.y;
    if (random <= bary_sum)
        return ivec2(ceil(pixel));

    bary_sum += relative_pos.x * (1. - relative_pos.y);
    if (random <= bary_sum)
        return ivec2(0, 1) * ivec2(floor(pixel)) + (1 - ivec2(0, 1)) * ivec2(ceil(pixel));

    bary_sum += (1. - relative_pos.x) * relative_pos.y;
    if (random <= bary_sum)
        return ivec2(1, 0) * ivec2(floor(pixel)) + (1 - ivec2(1, 0)) * ivec2(ceil(pixel));

    // (1, 1)
    return ivec2(floor(pixel));
}

#endif
