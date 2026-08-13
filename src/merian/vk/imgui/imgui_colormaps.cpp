#include "merian/vk/imgui/imgui_colormaps.hpp"

#include <algorithm>
#include <cmath>

namespace merian {

// Coefficients as in colors/colormaps.slang, so overlays and panels agree.
ImU32 imgui_colormap_turbo(float x) {
    x = std::clamp(x, 0.f, 1.f);
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float x4 = x2 * x2;
    const float x5 = x4 * x;
    const float r = 0.13572138f + (4.61539260f * x) - (42.66032258f * x2) + (132.13108234f * x3) -
                    (152.94239396f * x4) + (59.28637943f * x5);
    const float g = 0.09140261f + (2.19418839f * x) + (4.84296658f * x2) - (14.18503333f * x3) +
                    (4.27729857f * x4) + (2.82956604f * x5);
    const float b = 0.10667330f + (12.64194608f * x) - (60.58204836f * x2) + (110.36276771f * x3) -
                    (89.90310912f * x4) + (27.34824973f * x5);
    const auto to8 = [](const float v) {
        return static_cast<int>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f));
    };
    return IM_COL32(to8(r), to8(g), to8(b), 255);
}

ImU32 imgui_colormap_diverging(float x) {
    x = std::clamp(x, 0.f, 1.f);
    const auto mix = [](const float a, const float b, const float f) { return a + ((b - a) * f); };
    if (x < 0.5f) {
        const float f = x * 2.f;
        return IM_COL32(static_cast<int>(mix(25, 242, f)), static_cast<int>(mix(64, 242, f)),
                        static_cast<int>(mix(217, 242, f)), 255);
    }
    const float f = (x - 0.5f) * 2.f;
    return IM_COL32(static_cast<int>(mix(242, 217, f)), static_cast<int>(mix(242, 38, f)),
                    static_cast<int>(mix(242, 25, f)), 255);
}

void imgui_colorbar(ImDrawList* const dl,
                    const ImVec2 min,
                    const ImVec2 max,
                    ImU32 (*colormap)(float)) {
    constexpr int SEGMENTS = 16;
    const float w = max.x - min.x;
    for (int s = 0; s < SEGMENTS; s++) {
        const float t0 = static_cast<float>(s) / SEGMENTS;
        const float t1 = static_cast<float>(s + 1) / SEGMENTS;
        dl->AddRectFilledMultiColor(ImVec2(min.x + (t0 * w), min.y),
                                    ImVec2(min.x + (t1 * w), max.y), colormap(t0), colormap(t1),
                                    colormap(t1), colormap(t0));
    }
}

} // namespace merian
