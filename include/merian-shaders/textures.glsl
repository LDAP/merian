
#extension GL_GOOGLE_include_directive : require


#ifndef _MERIAN_SHADERS_TEXTURES_H_
#define _MERIAN_SHADERS_TEXTURES_H_

// adapted from gl_warp.c
#define MERIAN_TEXTUREEFFECT_QUAKE_WARPCALC(st, time) (2 * ((st) + 0.125 * sin(vec2(3 * (st).yx + 0.5 * (time)))))

#define MERIAN_TEXTUREEFFECT_WAVES(st, time) (-vec2(.1, 0) * cos((st).x * 5 + (time) * 2) * pow(max(sin((st).x * 5 + (time) * 2), 0), 5) \
                         -vec2(.07, 0) * cos(-(st).x * 5 + -(st).y * 3 + (time) * 3) * pow(max(sin(-(st).x * 5 + -(st).y * 3 + (time) * 3), 0), 5))

#define MERIAN_TEX_COORD_TO_PIXEL(tc, resolution) ((tc) * (resolution) - 0.5)

#define MERIAN_PIXEL_TO_TEX_COORD(pixel, resolution) (((pixel) + 0.5) / (resolution))



ivec2 merian_tex_clamp(const ivec2 ipos, const ivec2 tex_size) {
    return clamp(ipos, ivec2(0), tex_size - 1);
}

ivec2 merian_tex_repeat(const ivec2 ipos, const ivec2 tex_size) {
    return ipos % tex_size;
}

// mirror(a) returns a if a ≥ 0, and −(1 + a) otherwise
#define _MERIAN_MIRROR(ipos) mix(ipos, -(1 + ipos), lessThan(ipos, ivec2(0)))


ivec2 merian_tex_mirror_repeat(const ivec2 ipos, const ivec2 tex_size) {
    return tex_size - 1 - _MERIAN_MIRROR(ipos % (tex_size * 2) - tex_size);
}


#endif // _MERIAN_SHADERS_TEXTURES_H_
