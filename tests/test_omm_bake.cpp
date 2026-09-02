// Reference for the opacity micromap bake, and the properties the shader has to keep.
//
// `include/merian-shaders/scene/omm-bake.slang` mirrors these functions; they are kept here in a
// form that runs without a device so the curve and the rasterization can be checked against brute
// force rather than against a render.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct Vec2 {
    float x = 0.f;
    float y = 0.f;
};

Vec2 operator+(const Vec2 a, const Vec2 b) {
    return {a.x + b.x, a.y + b.y};
}
Vec2 operator-(const Vec2 a, const Vec2 b) {
    return {a.x - b.x, a.y - b.y};
}
Vec2 operator*(const Vec2 a, const float s) {
    return {a.x * s, a.y * s};
}

struct MicroTriangle {
    Vec2 p0;
    Vec2 p1;
    Vec2 p2;
};

// --- the curve ---

uint32_t extract_even_bits(uint32_t x) {
    x &= 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0f0f0f0f;
    x = (x | (x >> 4)) & 0x00ff00ff;
    x = (x | (x >> 8)) & 0x0000ffff;
    return x;
}

uint32_t prefix_eor(uint32_t x) {
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;
    return x;
}

// Barycentric corners of the micro-triangle at `index`, in the ordering the opacity micromap
// lookup uses (Vulkan, "Ray Opacity Micromap").
MicroTriangle index_to_bary(const uint32_t index, const uint32_t level) {
    if (level == 0) {
        return {{0, 0}, {1, 0}, {0, 1}};
    }

    const uint32_t b0 = extract_even_bits(index);
    const uint32_t b1 = extract_even_bits(index >> 1);
    const uint32_t fx = prefix_eor(b0);
    const uint32_t fy = prefix_eor(b0 & ~b1);
    const uint32_t t = fy ^ b1;

    const uint32_t mask = (1u << level) - 1;
    uint32_t iu = ((fx & ~t) | (b0 & ~t) | (~b0 & ~fx & t)) & mask;
    uint32_t iv = (fy ^ b0) & mask;
    const uint32_t iw = ((~fx & ~t) | (b0 & ~t) | (~b0 & fx & t)) & mask;

    const bool upright = ((iu & 1) ^ (iv & 1) ^ (iw & 1)) != 0;
    if (!upright) {
        iu += 1;
        iv += 1;
    }

    const float scale = 1.f / static_cast<float>(1u << level);
    const float step = upright ? scale : -scale;
    const float u = static_cast<float>(iu) * scale;
    const float v = static_cast<float>(iv) * scale;
    return {{u, v}, {u + step, v}, {u, v + step}};
}

// --- rasterization ---

float edge_area(const Vec2 a, const Vec2 b, const Vec2 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

bool box_inside_edge(const Vec2 a, const Vec2 b, const Vec2 corner, const Vec2 extent) {
    const Vec2 normal{b.y - a.y, a.x - b.x};
    const float offset = -(normal.x * a.x + normal.y * a.y);
    const float bx = normal.x > 0.f ? 0.f : normal.x;
    const float by = normal.y > 0.f ? 0.f : normal.y;
    return normal.x * corner.x + normal.y * corner.y + offset + bx * extent.x + by * extent.y < 0.f;
}

// The texel centres (integer lattice points) a bilinear lookup anywhere inside the triangle can
// read. Corners are in texel space, where a texel centre sits on an integer.
std::vector<std::pair<int, int>> covered_texels(const MicroTriangle& t, const int max_span) {
    const Vec2 lo{std::min({t.p0.x, t.p1.x, t.p2.x}), std::min({t.p0.y, t.p1.y, t.p2.y})};
    const Vec2 hi{std::max({t.p0.x, t.p1.x, t.p2.x}), std::max({t.p0.y, t.p1.y, t.p2.y})};

    const int first_x = static_cast<int>(std::floor(lo.x));
    const int first_y = static_cast<int>(std::floor(lo.y));
    const int last_x = static_cast<int>(std::floor(hi.x)) + 1;
    const int last_y = static_cast<int>(std::floor(hi.y)) + 1;
    if (last_x - first_x + 1 > max_span || last_y - first_y + 1 > max_span) {
        return {};
    }

    const bool flip = edge_area(t.p0, t.p1, t.p2) > 0.f;
    const Vec2 a = t.p0;
    const Vec2 b = flip ? t.p2 : t.p1;
    const Vec2 c = flip ? t.p1 : t.p2;
    const Vec2 extent{2.f, 2.f};

    std::vector<std::pair<int, int>> out;
    for (int y = first_y; y <= last_y; y++) {
        bool was_inside = false;
        for (int x = first_x; x <= last_x; x++) {
            const Vec2 corner{static_cast<float>(x) - 1.f, static_cast<float>(y) - 1.f};
            if (!box_inside_edge(a, b, corner, extent) || !box_inside_edge(b, c, corner, extent) ||
                !box_inside_edge(c, a, corner, extent)) {
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

// --- helpers for the checks ---

bool point_in_triangle(const MicroTriangle& t, const Vec2 p) {
    const float d0 = edge_area(t.p0, t.p1, p);
    const float d1 = edge_area(t.p1, t.p2, p);
    const float d2 = edge_area(t.p2, t.p0, p);
    return (d0 >= 0 && d1 >= 0 && d2 >= 0) || (d0 <= 0 && d1 <= 0 && d2 <= 0);
}

float triangle_area(const MicroTriangle& t) {
    return std::abs(edge_area(t.p0, t.p1, t.p2)) * 0.5f;
}

MicroTriangle to_texel_space(const MicroTriangle& bary, const MicroTriangle& uv, const float size) {
    const auto map = [&](const Vec2 b) {
        const Vec2 p = uv.p0 + (uv.p1 - uv.p0) * b.x + (uv.p2 - uv.p0) * b.y;
        return Vec2{p.x * size - 0.5f, p.y * size - 0.5f};
    };
    return {map(bary.p0), map(bary.p1), map(bary.p2)};
}

} // namespace

// Every micro-triangle has exactly the area its level implies, so the curve neither skips nor
// doubles up on any part of the triangle.
TEST(OmmBake, MicroTrianglesTileTheTriangle) {
    for (uint32_t level = 0; level <= 5; level++) {
        const uint32_t count = 1u << (2 * level);
        const float expected = 0.5f / static_cast<float>(count);

        double total = 0.0;
        for (uint32_t i = 0; i < count; i++) {
            const MicroTriangle t = index_to_bary(i, level);
            EXPECT_NEAR(triangle_area(t), expected, expected * 1e-3f)
                << "level " << level << " index " << i;
            total += triangle_area(t);
        }
        EXPECT_NEAR(total, 0.5, 1e-4) << "level " << level;
    }
}

// And they cover it without overlapping: a point inside the unit triangle falls in exactly one.
TEST(OmmBake, MicroTrianglesPartitionTheTriangle) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uniform(0.f, 1.f);

    for (uint32_t level = 1; level <= 4; level++) {
        const uint32_t count = 1u << (2 * level);
        std::vector<MicroTriangle> micro;
        micro.reserve(count);
        for (uint32_t i = 0; i < count; i++) {
            micro.push_back(index_to_bary(i, level));
        }

        for (int sample = 0; sample < 2000; sample++) {
            float u = uniform(rng);
            float v = uniform(rng);
            if (u + v > 1.f) {
                u = 1.f - u;
                v = 1.f - v;
            }
            // away from the shared edges, where a point legitimately lands on a boundary
            const Vec2 p{u, v};
            int hits = 0;
            for (const MicroTriangle& t : micro) {
                if (point_in_triangle(t, p)) {
                    hits++;
                }
            }
            EXPECT_GE(hits, 1) << "level " << level << " uncovered at " << u << "," << v;
        }
    }
}

// The rasterization must visit every texel a bilinear lookup inside the micro-triangle can read.
TEST(OmmBake, RasterCoversEveryReadableTexel) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> coord(2.f, 30.f);

    for (int trial = 0; trial < 200; trial++) {
        const MicroTriangle t{
            {coord(rng), coord(rng)}, {coord(rng), coord(rng)}, {coord(rng), coord(rng)}};
        if (triangle_area(t) < 1e-3f) {
            continue;
        }

        const auto visited = covered_texels(t, 1024);
        ASSERT_FALSE(visited.empty());

        // brute force: dense points inside the triangle, each reads the four lattice points around
        for (int i = 0; i <= 60; i++) {
            for (int j = 0; j + i <= 60; j++) {
                const float bu = static_cast<float>(i) / 60.f;
                const float bv = static_cast<float>(j) / 60.f;
                const Vec2 p = t.p0 + (t.p1 - t.p0) * bu + (t.p2 - t.p0) * bv;
                const int lx = static_cast<int>(std::floor(p.x));
                const int ly = static_cast<int>(std::floor(p.y));
                for (const auto [tx, ty] :
                     {std::pair{lx, ly}, {lx + 1, ly}, {lx, ly + 1}, {lx + 1, ly + 1}}) {
                    EXPECT_NE(std::find(visited.begin(), visited.end(), std::pair{tx, ty}),
                              visited.end())
                        << "texel " << tx << "," << ty << " readable but not visited";
                }
            }
        }
    }
}

// A state the bake commits to must hold everywhere on the micro-triangle: where it says opaque,
// no point inside may fail the alpha test, and the other way round.
TEST(OmmBake, CommittedStateHoldsEverywhere) {
    constexpr float size = 64.f;
    constexpr float threshold = 0.5f;
    // a disc of "opaque" texels, so micro-triangles land inside, outside and across the edge
    const auto alpha_at = [&](const int x, const int y) {
        const float dx = static_cast<float>(x) - 31.5f;
        const float dy = static_cast<float>(y) - 31.5f;
        return std::sqrt(dx * dx + dy * dy) < 18.f ? 1.f : 0.f;
    };
    const auto texel_passes = [&](const int x, const int y) {
        const int cx = std::clamp(x, 0, 63);
        const int cy = std::clamp(y, 0, 63);
        return alpha_at(cx, cy) >= threshold;
    };

    const MicroTriangle uv{{0.05f, 0.05f}, {0.95f, 0.1f}, {0.1f, 0.95f}};
    const uint32_t level = 4;
    const uint32_t count = 1u << (2 * level);

    int opaque = 0;
    int transparent = 0;
    int unknown = 0;
    for (uint32_t i = 0; i < count; i++) {
        const MicroTriangle t = to_texel_space(index_to_bary(i, level), uv, size);
        const auto visited = covered_texels(t, 1024);

        bool any_pass = false;
        bool any_fail = false;
        for (const auto [x, y] : visited) {
            (texel_passes(x, y) ? any_pass : any_fail) = true;
        }

        if (any_pass && any_fail) {
            unknown++;
            continue;
        }
        if (visited.empty()) {
            unknown++;
            continue;
        }

        // brute force every point of the micro-triangle against the bilinear footprint
        for (int a = 0; a <= 24; a++) {
            for (int b = 0; a + b <= 24; b++) {
                const float bu = static_cast<float>(a) / 24.f;
                const float bv = static_cast<float>(b) / 24.f;
                const Vec2 p = t.p0 + (t.p1 - t.p0) * bu + (t.p2 - t.p0) * bv;
                const int lx = static_cast<int>(std::floor(p.x));
                const int ly = static_cast<int>(std::floor(p.y));
                for (const auto [tx, ty] :
                     {std::pair{lx, ly}, {lx + 1, ly}, {lx, ly + 1}, {lx + 1, ly + 1}}) {
                    if (any_pass) {
                        EXPECT_TRUE(texel_passes(tx, ty)) << "committed opaque, but a reachable "
                                                             "texel fails the alpha test";
                    } else {
                        EXPECT_FALSE(texel_passes(tx, ty)) << "committed transparent, but a "
                                                              "reachable texel passes";
                    }
                }
            }
        }
        (any_pass ? opaque : transparent)++;
    }

    // the point of the bake: most of a cutout is settled either way
    EXPECT_GT(opaque + transparent, static_cast<int>(count) / 2)
        << "opaque " << opaque << " transparent " << transparent << " unknown " << unknown;
}

// Whatever the uvs do, one micro-triangle costs a bounded number of texels.
TEST(OmmBake, WorkStaysBounded) {
    constexpr int max_span = 64;
    const MicroTriangle huge{{-1e6f, -1e6f}, {1e6f, -1e6f}, {-1e6f, 1e6f}};
    EXPECT_TRUE(covered_texels(huge, max_span).empty());

    const MicroTriangle degenerate{{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
    EXPECT_LE(covered_texels(degenerate, max_span).size(), 4u);

    std::mt19937 rng(3);
    std::uniform_real_distribution<float> coord(-40.f, 40.f);
    for (int trial = 0; trial < 500; trial++) {
        const MicroTriangle t{
            {coord(rng), coord(rng)}, {coord(rng), coord(rng)}, {coord(rng), coord(rng)}};
        EXPECT_LE(covered_texels(t, max_span).size(),
                  static_cast<size_t>(max_span + 1) * (max_span + 1));
    }
}
