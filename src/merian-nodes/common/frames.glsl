#ifndef _FRAMES_H_
#define _FRAMES_H_

// Returns a matrix of axis (x, y, z) which all being perpendicular to each other
mat3x3 make_frame(vec3 z) {
    const vec3 up = (abs(z.x) > abs(z.y)) ? vec3(0,1,0) : vec3(1,0,0);
    const vec3 du = normalize(cross(up, z));
    return mat3(du, normalize(cross(du, z)), z);
}

#endif
