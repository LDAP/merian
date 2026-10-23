#include "merian/vk/imgui/imgui_colormaps.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace merian {

namespace {

int to8(const float v) {
    return static_cast<int>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f));
}

} // namespace

// Coefficients as in merian-shaders/colors/colormaps.slang.
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
    return IM_COL32(to8(r), to8(g), to8(b), 255);
}

ImU32 imgui_colormap_plasma(float x) {
    x = std::clamp(x, 0.f, 1.f);
    const auto poly = [x](const std::array<float, 7>& c) {
        float v = c[6];
        for (int i = 5; i >= 0; i--) {
            v = c[i] + (x * v);
        }
        return to8(v);
    };
    return IM_COL32(
        poly({0.05873234392399702f, 2.176514634195958f, -2.689460476458034f, 6.130348345893603f,
              -11.10743619062271f, 10.02306557647065f, -3.658713842777788f}),
        poly({0.02333670892565664f, 0.2383834171260182f, -7.455851135738909f, 42.3461881477227f,
              -82.66631109428045f, 71.41361770095349f, -22.93153465461149f}),
        poly({0.5433401826748754f, 0.7539604599784036f, 3.110799939717086f, -28.51885465332158f,
              60.13984767418263f, -54.07218655560067f, 18.19190778539828f}),
        255);
}

ImU32 imgui_colormap_diverging(float x) {
    x = std::clamp(x, 0.f, 1.f);
    constexpr std::array<float, 3> LO = {0.10f, 0.25f, 0.85f};
    constexpr std::array<float, 3> MID = {0.95f, 0.95f, 0.95f};
    constexpr std::array<float, 3> HI = {0.85f, 0.15f, 0.10f};
    const bool low = x < 0.5f;
    const float f = low ? x * 2.f : (x - 0.5f) * 2.f;
    const auto channel = [&](const std::size_t c) {
        const float a = low ? LO.at(c) : MID.at(c);
        const float b = low ? MID.at(c) : HI.at(c);
        return to8(a + ((b - a) * f));
    };
    return IM_COL32(channel(0), channel(1), channel(2), 255);
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
