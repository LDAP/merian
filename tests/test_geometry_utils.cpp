#include <gtest/gtest.h>

#include "merian/utils/aabb.hpp"
#include "merian/utils/colors.hpp"
#include "merian/utils/math.hpp"

#include <array>

using namespace merian;

TEST(GeometryUtils, WrapIsNonNegative) {
    EXPECT_EQ(wrap(5, 4), 1u);
    EXPECT_EQ(wrap(-1, 4), 3u);
    EXPECT_EQ(wrap(-8, 4), 0u);
}

TEST(GeometryUtils, TriangleArea) {
    EXPECT_FLOAT_EQ(triangle_area(float3(0, 0, 0), float3(2, 0, 0), float3(0, 3, 0)), 3.f);
    EXPECT_FLOAT_EQ(triangle_area(float3(1, 1, 1), float3(1, 1, 1), float3(0, 3, 0)), 0.f);
}

TEST(GeometryUtils, ConvexPolygonContainsEitherWinding) {
    const std::array<float2, 4> ccw = {float2(0, 0), float2(2, 0), float2(2, 2), float2(0, 2)};
    const std::array<float2, 4> cw = {float2(0, 0), float2(0, 2), float2(2, 2), float2(2, 0)};
    for (const auto& polygon : {ccw, cw}) {
        EXPECT_TRUE(convex_polygon_contains(polygon, float2(1, 1)));
        EXPECT_FALSE(convex_polygon_contains(polygon, float2(3, 1)));
        EXPECT_FALSE(convex_polygon_contains(polygon, float2(-0.5f, 1)));
    }
}

TEST(GeometryUtils, AABBDistanceSq) {
    const AABB box(float3(0), float3(1));
    EXPECT_FLOAT_EQ(box.distance_sq(float3(0.5f)), 0.f);
    EXPECT_FLOAT_EQ(box.distance_sq(float3(3, 0.5f, 0.5f)), 4.f);
    EXPECT_FLOAT_EQ(box.distance_sq(float3(-1, -1, 0.5f)), 2.f);
}

TEST(GeometryUtils, SrgbToLinear) {
    EXPECT_FLOAT_EQ(srgb_to_linear(0.f), 0.f);
    EXPECT_NEAR(srgb_to_linear(1.f), 1.f, 1e-6f);
    EXPECT_NEAR(srgb_to_linear(0.5f), 0.2140411f, 1e-6f);
}
