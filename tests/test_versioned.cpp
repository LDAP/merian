#include <gtest/gtest.h>

#include "merian/utils/versioned.hpp"

using namespace merian;

static std::shared_ptr<int> boxed(int v) {
    return std::make_shared<int>(v);
}

// A burst of input changes collapses to a single rebuild on the next pull.
TEST(Versioned, CoalescesChangesUntilPulled) {
    uint64_t input_version = 0;
    int input = 0;

    uint32_t build_count = 0;
    auto derived = Versioned<int>([&] {
        build_count++;
        return boxed(input * 2);
    });
    derived.depends_on([&] { return input_version; });

    EXPECT_EQ(build_count, 0u) << "must not build eagerly";

    for (int i = 1; i <= 5; i++) {
        input = i;
        input_version++;
    }
    EXPECT_EQ(build_count, 0u) << "changes don't build, only the next get() does";

    EXPECT_EQ(*derived.get(), 10);
    EXPECT_EQ(build_count, 1u) << "five changes collapse into one rebuild";

    EXPECT_EQ(*derived.get(), 10);
    EXPECT_EQ(build_count, 1u) << "unchanged inputs: no rebuild";
}

// A value shared by two consumers (diamond) rebuilds at most once per pull.
TEST(Versioned, DiamondRebuildsSharedValueOnce) {
    uint64_t input_version = 0;
    int input = 1;

    uint32_t shared_builds = 0;
    auto shared = Versioned<int>([&] {
        shared_builds++;
        return boxed(input + 100);
    });
    shared.depends_on([&] { return input_version; });

    auto left = Versioned<int>([&] { return boxed(*shared.get() + 1); }).depends_on(shared);
    auto right = Versioned<int>([&] { return boxed(*shared.get() + 2); }).depends_on(shared);

    input = 5;
    input_version++;

    EXPECT_EQ(*left.get(), 106);
    EXPECT_EQ(*right.get(), 107);
    EXPECT_EQ(shared_builds, 1u) << "shared input rebuilds once for both consumers";
}

// A pulled value keeps its contents after the input moves on.
TEST(Versioned, PulledValueStaysValidAfterRebuild) {
    uint64_t input_version = 0;
    int input = 7;
    auto derived = Versioned<int>([&] { return boxed(input); });
    derived.depends_on([&] { return input_version; });

    auto pinned = derived.get();
    EXPECT_EQ(*pinned, 7);

    input = 42;
    input_version++;
    EXPECT_EQ(*derived.get(), 42);
    EXPECT_EQ(*pinned, 7) << "old value is frozen";
}

// peek() never triggers a build.
TEST(Versioned, PeekDoesNotBuild) {
    uint64_t input_version = 0;
    uint32_t build_count = 0;
    auto derived = Versioned<int>([&] {
        build_count++;
        return boxed(3);
    });
    derived.depends_on([&] { return input_version; });

    EXPECT_EQ(derived.peek(), nullptr);
    EXPECT_EQ(build_count, 0u);

    derived.get();
    EXPECT_EQ(build_count, 1u);
    input_version++;
    EXPECT_NE(derived.peek(), nullptr) << "peek returns the stale value, still no build";
    EXPECT_EQ(build_count, 1u);
}

// A chained dependency rebuilds the whole chain on pull.
TEST(Versioned, DependencyChainRebuilds) {
    uint64_t input_version = 0;
    int input = 1;

    auto a = Versioned<int>([&] { return boxed(input); });
    a.depends_on([&] { return input_version; });
    auto b = Versioned<int>([&] { return boxed(*a.get() + 10); }).depends_on(a);

    EXPECT_EQ(*b.get(), 11);

    input = 2;
    input_version++;
    EXPECT_EQ(*b.get(), 12) << "change to a's input propagates through to b";
}

// A value with no inputs builds exactly once.
TEST(Versioned, NoInputsBuildsOnce) {
    uint32_t build_count = 0;
    auto derived = Versioned<int>([&] {
        build_count++;
        return boxed(1);
    });
    derived.get();
    derived.get();
    EXPECT_EQ(build_count, 1u);
}
