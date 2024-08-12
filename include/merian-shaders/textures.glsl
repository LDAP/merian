#ifndef _TEXTURES_H_
#define _TEXTURES_H_

// adapted from gl_warp.c
#define MERIAN_TEXTUREEFFECT_QUAKE_WARPCALC(st, time) (2 * ((st) + 0.125 * sin(vec2(3 * (st).yx + 0.5 * (time)))))

#define MERIAN_TEXTUREEFFECT_WAVES(st, time) (-vec2(.1, 0) * cos((st).x * 5 + (time) * 2) * pow(max(sin((st).x * 5 + (time) * 2), 0), 5) \
                         -vec2(.07, 0) * cos(-(st).x * 5 + -(st).y * 3 + (time) * 3) * pow(max(sin(-(st).x * 5 + -(st).y * 3 + (time) * 3), 0), 5))

#endif // _TEXTURES_H_
