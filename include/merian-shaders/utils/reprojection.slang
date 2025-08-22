#pragma once

namespace merian {

// Computes a weight for the quality of the reprojection.
// Reprojections with normals further away than normal_reject_cos = cos(alpha)
// and depths farther apart that z_reject_percent * max(curr_z, prev_z) are completely rejected
float reprojection_weight(float3 curr_normal,
                          float3 prev_normal,
                          float normal_reject_cos,
                          float curr_z,
                          float vel_z,
                          float prev_z,
                          float z_reject_percent) {
    return smoothstep(normal_reject_cos, 1.0f, dot(curr_normal, prev_normal)) *
           (1.0f - smoothstep(0.0f, z_reject_percent * max(curr_z, prev_z),
                              abs((curr_z - prev_z) + vel_z)));
}

float reprojection_weight(float3 curr_normal,
                          float3 prev_normal,
                          float normal_reject_cos,
                          float curr_z,
                          float vel_z,
                          int2 pixel_offset,
                          float2 grad_z,
                          float prev_z,
                          float z_reject_percent) {
    return smoothstep(normal_reject_cos, 1.0f, dot(curr_normal, prev_normal)) *
           (1.0f -
            smoothstep(0.0f, z_reject_percent * max(curr_z, prev_z),
                       abs(((curr_z + dot(float2(pixel_offset), grad_z)) - prev_z) + vel_z)));
}

// Like reprojection_weight but binary
bool reprojection_valid(float3 curr_normal,
                        float3 prev_normal,
                        float normal_reject_cos,
                        float curr_z,
                        float vel_z,
                        int2 pixel_offset,
                        float2 grad_z,
                        float prev_z,
                        float z_reject_percent) {
    bool _109 = normal_reject_cos <= dot(curr_normal, prev_normal);
    bool _121;
    if (_109) {
        _121 = abs(((curr_z + dot(float2(pixel_offset), grad_z)) - prev_z) + vel_z) <=
               (z_reject_percent * max(curr_z, prev_z));
    } else {
        _121 = _109;
    }
    return _121;
}

bool reprojection_valid(float3 curr_normal,
                        float3 prev_normal,
                        float normal_reject_cos,
                        float curr_z,
                        float vel_z,
                        float prev_z,
                        float z_reject_percent) {
    bool _124 = normal_reject_cos <= dot(curr_normal, prev_normal);
    bool _133;
    if (_124) {
        _133 = abs((curr_z - prev_z) + vel_z) <= (z_reject_percent * max(curr_z, prev_z));
    } else {
        _133 = _124;
    }
    return _133;
}

// Intersects the motion vector with the image borders.
//
// Returns true if prev pos was outside the image borders.
// prev_pos = pos + mv
bool reprojection_intersect_border(float2 prev_pos, float2 mv, float2 image_size_minus_one) {
    float2 _136 = round(prev_pos);
    bool _138 = any(bool2(_136.x > image_size_minus_one.x, _136.y > image_size_minus_one.y));
    bool _146;
    if (!_138) {
        float2 _143 = round(prev_pos);
        _146 = any(bool2(_143.x < 0.0f.xx.x, _143.y < 0.0f.xx.y));
    } else {
        _146 = _138;
    }
    if (_146) {
        float tmin = max(min(prev_pos.x / mv.x, (prev_pos.x - image_size_minus_one.x) / mv.x),
                         min(prev_pos.y / mv.y, (prev_pos.y - image_size_minus_one.y) / mv.y));
        prev_pos -= (mv * tmin);
        return true;
    }
    return false;
}

// pixel = current_pixel + mv
int2 reproject_pixel_nearest(float2 pixel) {
    return int2(round(pixel));
}

// Performs stochastic bilinear interpolation when reprojecting
// pixel = current_pixel + mv
int2 reproject_pixel_stochastic(float2 pixel, float random) {
    float2 relative_pos = frac(pixel);
    float bary_sum = relative_pos.x * relative_pos.y;
    if (random <= bary_sum) {
        return int2(ceil(pixel));
    }
    bary_sum += (relative_pos.x * (1.0f - relative_pos.y));
    if (random <= bary_sum) {
        return (int2(0, 1) * int2(floor(pixel))) + (int2(1, 0) * int2(ceil(pixel)));
    }
    bary_sum += ((1.0f - relative_pos.x) * relative_pos.y);
    if (random <= bary_sum) {
        return (int2(1, 0) * int2(floor(pixel))) + (int2(0, 1) * int2(ceil(pixel)));
    }
    return int2(floor(pixel));
}

}
