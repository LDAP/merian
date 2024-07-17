#ifndef _MORTON_H_
#define _MORTON_H_

#include "common/bit_twiddling.glsl"

// Works only if width is power of two
// https://fgiesen.wordpress.com/2022/09/09/morton-codes-addendum/
uint morton_encode2d(in uvec2 p) {
    p.x = p.x & 0x0000ffff;                    // p.x = ---- ---- ---- ---- fedc ba98 7654 3210
    p.x = (p.x ^ (p.x <<  8)) & 0x00ff00ff;    // p.x = ---- ---- fedc ba98 ---- ---- 7654 3210
    p.x = (p.x ^ (p.x <<  4)) & 0x0f0f0f0f;    // p.x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    p.x = (p.x ^ (p.x <<  2)) & 0x33333333;    // p.x = --fe --dc --ba --98 --76 --54 --32 --10
    p.x = (p.x ^ (p.x <<  1)) & 0x55555555;    // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

    p.y = p.y & 0x0000ffff;                    // p.y = ---- ---- ---- ---- fedc ba98 7654 3210
    p.y = (p.y ^ (p.y <<  8)) & 0x00ff00ff;    // p.y = ---- ---- fedc ba98 ---- ---- 7654 3210
    p.y = (p.y ^ (p.y <<  4)) & 0x0f0f0f0f;    // p.y = ---- fedc ---- ba98 ---- 7654 ---- 3210
    p.y = (p.y ^ (p.y <<  2)) & 0x33333333;    // p.y = --fe --dc --ba --98 --76 --54 --32 --10
    p.y = (p.y ^ (p.y <<  1)) & 0x55555555;    // p.y = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

    return p.x | (p.y << 1);
}

uint morton_encode2d(const uvec2 p, in uint width, in uint height) {
    width = next_power_of_two(width);
    height = next_power_of_two(height);
    const uint msb = min(findMSB(width), findMSB(height));
    const uint zorder_mask = (1 << (msb << 1)) - 1;
    // right side is linear part
    return (morton_encode2d(p) & zorder_mask) | (((p.x | p.y) << msb) & (~zorder_mask));
}


#endif // _MORTON_H_
