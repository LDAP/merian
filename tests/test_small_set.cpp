#include <gtest/gtest.h>

#include "merian/utils/small_set.hpp"

#include <algorithm>
#include <map>
#include <vector>

using namespace merian;

TEST(SmallSet, EmptyOnConstruction) {
    SmallSet<int, 4> s;
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.size(), 0u);
    EXPECT_FALSE(s.contains(0));
}

TEST(SmallSet, InsertKeepsSorted) {
    SmallSet<int, 8> s;
    for (int v : {5, 1, 9, 3, 7})
        s.insert(v);
    ASSERT_EQ(s.size(), 5u);
    EXPECT_TRUE(std::is_sorted(s.begin(), s.end()));
    EXPECT_EQ(s.front(), 1);
}

TEST(SmallSet, InsertRejectsDuplicates) {
    SmallSet<int, 4> s;
    auto [it1, inserted1] = s.insert(42);
    auto [it2, inserted2] = s.insert(42);
    EXPECT_TRUE(inserted1);
    EXPECT_FALSE(inserted2);
    EXPECT_EQ(*it1, 42);
    EXPECT_EQ(*it2, 42);
    EXPECT_EQ(s.size(), 1u);
}

TEST(SmallSet, InsertSpillsToHeap) {
    SmallSet<int, 1> s;
    s.insert(2);
    s.insert(1); // sorts before, forces spill
    s.insert(3);
    s.insert(0);
    ASSERT_EQ(s.size(), 4u);
    EXPECT_TRUE(std::is_sorted(s.begin(), s.end()));
    EXPECT_EQ(s.front(), 0);
}

TEST(SmallSet, EraseExistingAndMissing) {
    SmallSet<int, 4> s;
    for (int v : {1, 2, 3, 4})
        s.insert(v);
    EXPECT_EQ(s.erase(3), 1u);
    EXPECT_EQ(s.size(), 3u);
    EXPECT_FALSE(s.contains(3));

    EXPECT_EQ(s.erase(99), 0u); // missing
    EXPECT_EQ(s.size(), 3u);
}

TEST(SmallSet, EraseAfterSpill) {
    SmallSet<int, 1> s;
    for (int v : {10, 20, 30, 40})
        s.insert(v);
    EXPECT_EQ(s.erase(20), 1u);
    ASSERT_EQ(s.size(), 3u);
    EXPECT_TRUE(std::is_sorted(s.begin(), s.end()));
    EXPECT_EQ(s.front(), 10);
}

TEST(SmallSet, Contains) {
    SmallSet<int, 2> s;
    for (int v : {1, 2, 3, 4, 5})
        s.insert(v);
    for (int v : {1, 3, 5})
        EXPECT_TRUE(s.contains(v));
    for (int v : {0, 6, -1})
        EXPECT_FALSE(s.contains(v));
}

TEST(SmallSet, CountMatchesStdSet) {
    SmallSet<int, 2> s;
    for (int v : {1, 2, 3})
        s.insert(v);
    EXPECT_EQ(s.count(2), 1u);
    EXPECT_EQ(s.count(99), 0u);
}

TEST(SmallSet, EqualityAndOrderingAsMapKey) {
    SmallSet<int, 1> a;
    a.insert(1);
    a.insert(2);
    a.insert(3);

    SmallSet<int, 1> b;
    // Insert out of order — sorted invariant means a == b.
    b.insert(3);
    b.insert(1);
    b.insert(2);
    EXPECT_TRUE(a == b);

    SmallSet<int, 1> c;
    c.insert(1);
    c.insert(2);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(c < a); // shorter prefix sorts first

    // Use as a std::map key.
    std::map<SmallSet<int, 1>, int> m;
    m[a] = 100;
    m[c] = 200;
    EXPECT_EQ(m.size(), 2u);
    EXPECT_EQ(m[b], 100); // same key as a
    EXPECT_EQ(m[c], 200);
}

TEST(SmallSet, RangeForIteration) {
    SmallSet<int, 2> s;
    for (int v : {5, 3, 1, 4, 2})
        s.insert(v);
    int prev = 0;
    int seen = 0;
    for (int v : s) {
        EXPECT_GT(v, prev);
        prev = v;
        ++seen;
    }
    EXPECT_EQ(seen, 5);
}

TEST(SmallSet, Clear) {
    SmallSet<int, 2> s;
    for (int v : {1, 2, 3, 4})
        s.insert(v);
    s.clear();
    EXPECT_TRUE(s.empty());
    EXPECT_FALSE(s.contains(1));
    s.insert(7);
    EXPECT_EQ(s.size(), 1u);
    EXPECT_EQ(s.front(), 7);
}
