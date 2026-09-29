
#include "merian-shaders/scene/omm-bake.slangh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

using namespace merian;

namespace {

std::vector<std::pair<int, int>> covered_texels(const OmmMicroTriangle& t) {
    const OmmTexelFootprint footprint = omm_texel_footprint(t.p0, t.p1, t.p2);
    if (!footprint.bounded) {
        return {};
    }

    std::vector<std::pair<int, int>> out;
    for (int y = footprint.first.y; y <= footprint.last.y; y++) {
        bool was_inside = false;
        for (int x = footprint.first.x; x <= footprint.last.x; x++) {
            if (!omm_footprint_reads(footprint, x, y)) {
                if (was_inside) {
                    break;
                }
                continue;
            }
            was_inside = true;
            out.emplace_back(x, y);
        }
    }
    return out;
}

bool point_in_triangle(const OmmMicroTriangle& t, const float2 p) {
    const float d0 = omm_edge_area(t.p0, t.p1, p);
    const float d1 = omm_edge_area(t.p1, t.p2, p);
    const float d2 = omm_edge_area(t.p2, t.p0, p);
    return (d0 >= 0 && d1 >= 0 && d2 >= 0) || (d0 <= 0 && d1 <= 0 && d2 <= 0);
}

float triangle_area(const OmmMicroTriangle& t) {
    return std::abs(omm_edge_area(t.p0, t.p1, t.p2)) * 0.5f;
}

OmmMicroTriangle map(const OmmMicroTriangle& bary, const OmmMicroTriangle& t) {
    const auto at = [&](const float2 b) {
        return t.p0 + (t.p1 - t.p0) * b.x + (t.p2 - t.p0) * b.y;
    };
    return {at(bary.p0), at(bary.p1), at(bary.p2)};
}

bool contains(const std::vector<std::pair<int, int>>& texels, const int x, const int y) {
    return std::find(texels.begin(), texels.end(), std::pair{x, y}) != texels.end();
}

std::vector<std::pair<int, int>> texels_read_near(const float2 p) {
    const float slack = 0.99f * static_cast<float>(MERIAN_OMM_TEXEL_SLACK);
    std::vector<std::pair<int, int>> out;
    for (const float dx : {-slack, 0.f, slack}) {
        for (const float dy : {-slack, 0.f, slack}) {
            const int lx = static_cast<int>(std::floor(p.x + dx));
            const int ly = static_cast<int>(std::floor(p.y + dy));
            out.insert(out.end(), {{lx, ly}, {lx + 1, ly}, {lx, ly + 1}, {lx + 1, ly + 1}});
        }
    }
    return out;
}

} // namespace

TEST(OmmBake, MicroTrianglesTileTheTriangle) {
    for (uint32_t level = 0; level <= 5; level++) {
        const uint32_t count = 1u << (2 * level);
        const float expected = 0.5f / static_cast<float>(count);

        double total = 0.0;
        for (uint32_t i = 0; i < count; i++) {
            const OmmMicroTriangle t = omm_index_to_bary(i, level);
            EXPECT_NEAR(triangle_area(t), expected, expected * 1e-3f)
                << "level " << level << " index " << i;
            total += triangle_area(t);
        }
        EXPECT_NEAR(total, 0.5, 1e-4) << "level " << level;
    }
}

TEST(OmmBake, MicroTrianglesCoverTheTriangle) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uniform(0.f, 1.f);

    for (uint32_t level = 1; level <= 4; level++) {
        const uint32_t count = 1u << (2 * level);
        std::vector<OmmMicroTriangle> micro;
        micro.reserve(count);
        for (uint32_t i = 0; i < count; i++) {
            micro.push_back(omm_index_to_bary(i, level));
        }

        for (int sample = 0; sample < 2000; sample++) {
            float u = uniform(rng);
            float v = uniform(rng);
            if (u + v > 1.f) {
                u = 1.f - u;
                v = 1.f - v;
            }
            const float2 p{u, v};
            EXPECT_TRUE(
                std::any_of(micro.begin(), micro.end(),
                            [&](const OmmMicroTriangle& t) { return point_in_triangle(t, p); }))
                << "level " << level << " uncovered at " << u << "," << v;
        }
    }
}

TEST(OmmBake, RasterCoversEveryReadableTexel) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> coord(2.f, 30.f);

    for (int trial = 0; trial < 200; trial++) {
        const OmmMicroTriangle t{
            {coord(rng), coord(rng)}, {coord(rng), coord(rng)}, {coord(rng), coord(rng)}};
        if (triangle_area(t) < 1e-3f) {
            continue;
        }

        const auto visited = covered_texels(t);
        ASSERT_FALSE(visited.empty());

        for (int i = 0; i <= 60; i++) {
            for (int j = 0; j + i <= 60; j++) {
                const float2 b{static_cast<float>(i) / 60.f, static_cast<float>(j) / 60.f};
                const float2 p = t.p0 + (t.p1 - t.p0) * b.x + (t.p2 - t.p0) * b.y;
                for (const auto [x, y] : texels_read_near(p)) {
                    EXPECT_TRUE(contains(visited, x, y))
                        << "texel " << x << "," << y << " readable but not visited";
                }
            }
        }
    }
}

TEST(OmmBake, CommittedStateHoldsEverywhere) {
    constexpr float size = 64.f;
    const auto texel_passes = [&](const int x, const int y) {
        const float dx = static_cast<float>(std::clamp(x, 0, 63)) - 31.5f;
        const float dy = static_cast<float>(std::clamp(y, 0, 63)) - 31.5f;
        return std::sqrt(dx * dx + dy * dy) < 18.f;
    };

    const OmmMicroTriangle uv{{0.05f, 0.05f}, {0.95f, 0.1f}, {0.1f, 0.95f}};
    const OmmMicroTriangle texel_space{uv.p0 * size - 0.5f, uv.p1 * size - 0.5f,
                                       uv.p2 * size - 0.5f};
    const uint32_t level = 4;
    const uint32_t count = 1u << (2 * level);

    int committed = 0;
    for (uint32_t i = 0; i < count; i++) {
        const OmmMicroTriangle t = map(omm_index_to_bary(i, level), texel_space);
        const auto visited = covered_texels(t);

        bool any_pass = false;
        bool any_fail = false;
        for (const auto [x, y] : visited) {
            (texel_passes(x, y) ? any_pass : any_fail) = true;
        }
        if (visited.empty() || (any_pass && any_fail)) {
            continue;
        }
        committed++;

        for (int a = 0; a <= 24; a++) {
            for (int b = 0; a + b <= 24; b++) {
                const float2 bary{static_cast<float>(a) / 24.f, static_cast<float>(b) / 24.f};
                const float2 p = t.p0 + (t.p1 - t.p0) * bary.x + (t.p2 - t.p0) * bary.y;
                for (const auto [x, y] : texels_read_near(p)) {
                    EXPECT_EQ(texel_passes(x, y), any_pass)
                        << "committed to one answer, but a reachable texel gives the other";
                }
            }
        }
    }

    EXPECT_GT(committed, static_cast<int>(count) / 2);
}

TEST(OmmBake, TriangleTexelsCoverItsMicroTriangles) {
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> coord(4.f, 40.f);

    for (int trial = 0; trial < 100; trial++) {
        const OmmMicroTriangle t{
            {coord(rng), coord(rng)}, {coord(rng), coord(rng)}, {coord(rng), coord(rng)}};
        if (triangle_area(t) < 1.f) {
            continue;
        }
        const auto whole = covered_texels(t);
        ASSERT_FALSE(whole.empty());

        for (uint32_t level = 1; level <= 3; level++) {
            const uint32_t count = 1u << (2 * level);
            for (uint32_t i = 0; i < count; i++) {
                for (const auto [x, y] : covered_texels(map(omm_index_to_bary(i, level), t))) {
                    EXPECT_TRUE(contains(whole, x, y))
                        << "micro-triangle " << i << " at level " << level
                        << " reads a texel its triangle does not";
                }
            }
        }
    }
}

TEST(OmmBake, WorkStaysBounded) {
    const OmmMicroTriangle huge{{-1e6f, -1e6f}, {1e6f, -1e6f}, {-1e6f, 1e6f}};
    EXPECT_FALSE(omm_texel_footprint(huge.p0, huge.p1, huge.p2).bounded);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(omm_texel_footprint({nan, 0.f}, {1.f, 0.f}, {0.f, 1.f}).bounded);
    EXPECT_FALSE(omm_texel_footprint({inf, inf}, {inf, inf}, {inf, inf}).bounded);
    EXPECT_FALSE(omm_texel_footprint({3e9f, 3e9f}, {3e9f, 3e9f}, {3e9f, 3e9f}).bounded);

    const OmmMicroTriangle degenerate{{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
    EXPECT_LE(covered_texels(degenerate).size(), 4u);

    std::mt19937 rng(3);
    std::uniform_real_distribution<float> coord(-200.f, 200.f);
    for (int trial = 0; trial < 500; trial++) {
        const float2 p0{coord(rng), coord(rng)};
        const float2 p1{coord(rng), coord(rng)};
        const float2 p2{coord(rng), coord(rng)};
        const OmmTexelFootprint footprint = omm_texel_footprint(p0, p1, p2);
        if (footprint.bounded) {
            EXPECT_LE(footprint.last.x - footprint.first.x + 1, MERIAN_OMM_MAX_SPAN);
            EXPECT_LE(footprint.last.y - footprint.first.y + 1, MERIAN_OMM_MAX_SPAN);
        }
    }
}
