#include <gtest/gtest.h>

#include "merian/utils/small_vector.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

using namespace merian;

TEST(SmallVector, EmptyOnConstruction) {
    SmallVector<int, 4> v;
    EXPECT_TRUE(v.empty());
    EXPECT_EQ(v.size(), 0u);
    EXPECT_EQ(v.begin(), v.end());
}

TEST(SmallVector, PushBackStaysInline) {
    SmallVector<int, 4> v;
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);
    EXPECT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 10);
    EXPECT_EQ(v[1], 20);
    EXPECT_EQ(v[2], 30);
    EXPECT_EQ(v.front(), 10);
}

TEST(SmallVector, PushBackSpillsToHeap) {
    SmallVector<int, 2> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3); // triggers spill
    v.push_back(4);
    EXPECT_EQ(v.size(), 4u);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
    EXPECT_EQ(v[3], 4);
}

TEST(SmallVector, InsertAtFrontMiddleEndInline) {
    SmallVector<int, 8> v;
    v.push_back(1);
    v.push_back(3);
    v.insert(v.begin() + 1, 2); // middle
    v.insert(v.begin(), 0);     // front
    v.insert(v.end(), 4);       // end
    ASSERT_EQ(v.size(), 5u);
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(v[i], i);
}

TEST(SmallVector, InsertCrossingSpillBoundary) {
    SmallVector<int, 2> v;
    v.push_back(0);
    v.push_back(2);
    // size == N: insert in middle should spill and place at idx 1
    v.insert(v.begin() + 1, 1);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 0);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 2);

    // Already on heap: continue inserting.
    v.insert(v.end(), 4);
    v.insert(v.begin() + 3, 3);
    ASSERT_EQ(v.size(), 5u);
    for (int i = 0; i < 5; i++)
        EXPECT_EQ(v[i], i);
}

TEST(SmallVector, EraseInline) {
    SmallVector<int, 4> v;
    for (int i : {10, 20, 30, 40})
        v.push_back(i);
    v.erase(v.begin() + 1);
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 10);
    EXPECT_EQ(v[1], 30);
    EXPECT_EQ(v[2], 40);

    v.erase(v.begin());
    v.erase(v.end() - 1);
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], 30);
}

TEST(SmallVector, EraseAfterSpill) {
    SmallVector<int, 2> v;
    for (int i = 0; i < 6; i++)
        v.push_back(i);
    v.erase(v.begin() + 3); // remove 3
    ASSERT_EQ(v.size(), 5u);
    EXPECT_EQ(v[0], 0);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 2);
    EXPECT_EQ(v[3], 4);
    EXPECT_EQ(v[4], 5);
}

TEST(SmallVector, RangeForIteration) {
    SmallVector<int, 2> v;
    for (int i = 1; i <= 5; i++)
        v.push_back(i);
    int sum = 0;
    for (int x : v)
        sum += x;
    EXPECT_EQ(sum, 1 + 2 + 3 + 4 + 5);
    EXPECT_EQ(std::accumulate(v.begin(), v.end(), 0), 15);
}

TEST(SmallVector, ClearReusesInline) {
    SmallVector<int, 2> v;
    for (int i = 0; i < 5; i++)
        v.push_back(i);
    v.clear();
    EXPECT_TRUE(v.empty());
    EXPECT_EQ(v.size(), 0u);

    // Refill from empty — should stay inline up to N again.
    v.push_back(42);
    EXPECT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], 42);
}

TEST(SmallVector, MoveOnlyType) {
    SmallVector<std::unique_ptr<int>, 1> v;
    v.push_back(std::make_unique<int>(7));
    v.push_back(std::make_unique<int>(11)); // forces spill
    v.push_back(std::make_unique<int>(13));
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(*v[0], 7);
    EXPECT_EQ(*v[1], 11);
    EXPECT_EQ(*v[2], 13);
}

TEST(SmallVector, CopyConstructAndAssign) {
    SmallVector<int, 2> a;
    for (int i = 0; i < 4; i++)
        a.push_back(i);

    SmallVector<int, 2> b = a;
    ASSERT_EQ(b.size(), a.size());
    EXPECT_TRUE(std::equal(a.begin(), a.end(), b.begin()));

    SmallVector<int, 2> c;
    c = a;
    ASSERT_EQ(c.size(), a.size());
    EXPECT_TRUE(std::equal(a.begin(), a.end(), c.begin()));

    // Mutating the copy must not affect the original.
    b.push_back(99);
    EXPECT_EQ(a.size(), 4u);
    EXPECT_EQ(b.size(), 5u);
}

TEST(SmallVector, InitListAssignmentReplacesContents) {
    SmallVector<int, 2> v;
    for (int i = 0; i < 5; i++)
        v.push_back(i);
    v = {42};
    ASSERT_EQ(v.size(), 1u);
    EXPECT_EQ(v[0], 42);

    v = {};
    EXPECT_TRUE(v.empty());

    v = {1, 2, 3};
    ASSERT_EQ(v.size(), 3u);
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 2);
    EXPECT_EQ(v[2], 3);
}

TEST(SmallVector, N1ZeroAllocCommonCase) {
    // Smoke test: with N=1, the common size-1 case must round-trip.
    SmallVector<int, 1> v;
    v.push_back(42);
    EXPECT_EQ(v.size(), 1u);
    EXPECT_EQ(v.front(), 42);
    EXPECT_EQ(v[0], 42);
    auto it = v.begin();
    EXPECT_EQ(*it, 42);
    EXPECT_EQ(++it, v.end());
}
