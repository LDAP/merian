#include "merian-graph/nodes/path_debug/layout.slangh"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {

using merian::path_debug_resample_key;
using merian::path_debug_unit_interval;

constexpr uint32_t TRIALS = 200000;

float uniform(std::mt19937& rng) {
    return path_debug_unit_interval(static_cast<uint32_t>(rng()));
}

std::vector<uint32_t> largest_keys(const std::vector<float>& keys, const uint32_t k) {
    std::vector<uint32_t> order(keys.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](const uint32_t a, const uint32_t b) { return keys[a] > keys[b]; });
    order.resize(k);
    std::sort(order.begin(), order.end());
    return order;
}

TEST(PathDebugResample, KeysStayFinite) {
    EXPECT_GT(path_debug_unit_interval(0u), 0.f);
    EXPECT_LT(path_debug_unit_interval(0xFFFFFFFFu), 1.f);
    EXPECT_TRUE(
        std::isfinite(path_debug_resample_key(1e30f, path_debug_unit_interval(0xFFFFFFFFu))));
}

TEST(PathDebugResample, KeysStayPositive) {
    for (const float u : {1.f, std::nextafter(1.f, 2.f)}) {
        const float key = path_debug_resample_key(1e30f, u);
        EXPECT_TRUE(std::isfinite(key)) << u;
        EXPECT_GT(key, 0.f) << u;
    }
}

TEST(PathDebugResample, LargestKeysSampleWithoutReplacement) {
    const std::vector<float> weights = {1.f, 2.f, 3.f, 4.f};
    const float total = std::accumulate(weights.begin(), weights.end(), 0.f);
    std::mt19937 rng(7);
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> drawn;
    std::vector<float> keys(weights.size());
    for (uint32_t trial = 0; trial < TRIALS; trial++) {
        for (std::size_t i = 0; i < weights.size(); i++) {
            keys[i] = path_debug_resample_key(weights[i], uniform(rng));
        }
        const std::vector<uint32_t> pair = largest_keys(keys, 2);
        drawn[{pair[0], pair[1]}]++;
    }

    for (uint32_t i = 0; i < weights.size(); i++) {
        for (uint32_t j = i + 1; j < weights.size(); j++) {
            const float expected = (weights[i] / total * weights[j] / (total - weights[i])) +
                                   (weights[j] / total * weights[i] / (total - weights[j]));
            const float measured = static_cast<float>(drawn[{i, j}]) / static_cast<float>(TRIALS);
            EXPECT_NEAR(measured, expected, 0.005f) << i << ", " << j;
        }
    }
}

TEST(PathDebugResample, AgedKeysWeighOlderPathsLess) {
    const float decay = 0.5f;
    std::mt19937 rng(11);
    uint32_t older = 0;
    std::vector<float> keys(4);
    for (uint32_t trial = 0; trial < TRIALS; trial++) {
        for (std::size_t i = 0; i < keys.size(); i++) {
            keys[i] = path_debug_resample_key(1.f, uniform(rng)) * (i < 2 ? decay : 1.f);
        }
        older += largest_keys(keys, 1)[0] < 2 ? 1 : 0;
    }
    EXPECT_NEAR(static_cast<float>(older) / static_cast<float>(TRIALS),
                2.f * decay / ((2.f * decay) + 2.f), 0.005f);
}

} // namespace
