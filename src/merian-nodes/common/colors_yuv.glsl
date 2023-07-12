#ifndef _COLORS_YUV_H_
#define _COLORS_YUV_H_

float yuv_luminance(const vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

#endif
