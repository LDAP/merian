#include <gtest/gtest.h>

#include "merian/utils/free_list.hpp"

#include <vector>

using namespace merian;

TEST(FreeList, AcquireExtendsAndReleaseAtTailCompacts) {
    FreeList<uint32_t> ids;
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_EQ(ids.count(), 0u);

    EXPECT_EQ(ids.acquire(), 0u);
    EXPECT_EQ(ids.acquire(), 1u);
    EXPECT_EQ(ids.acquire(), 2u);
    EXPECT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids.count(), 3u);

    EXPECT_TRUE(ids.release(2));
    EXPECT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids.count(), 2u);

    // Next acquire takes the new top, not a reused id.
    EXPECT_EQ(ids.acquire(), 2u);
    EXPECT_EQ(ids.size(), 3u);
}

TEST(FreeList, MidVectorReleaseGoesToPool) {
    FreeList<uint32_t> ids;
    for (int i = 0; i < 4; i++) {
        ids.acquire();
    }

    EXPECT_TRUE(ids.release(1));
    EXPECT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids.count(), 3u);
    EXPECT_FALSE(ids.is_used(1));

    // Pool reuses the freed slot before extending.
    EXPECT_EQ(ids.acquire(), 1u);
    EXPECT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids.count(), 4u);
}

TEST(FreeList, ReleasingTopAfterMidFreeCascadesCompaction) {
    FreeList<uint32_t> ids;
    for (int i = 0; i < 4; i++) {
        ids.acquire();
    }

    // Free 2 first (goes to pool), then 3 (top): both should compact away.
    ids.release(2);
    ids.release(3);
    EXPECT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids.count(), 2u);

    // Stale free_pool entry for `2` must be skipped on next acquire.
    EXPECT_EQ(ids.acquire(), 2u);
    EXPECT_EQ(ids.size(), 3u);
}

TEST(FreeList, AcquireSpecificIdReturnsBoolAndExtends) {
    FreeList<uint32_t> ids;

    EXPECT_TRUE(ids.acquire(static_cast<uint32_t>(5)));
    EXPECT_EQ(ids.size(), 6u);
    EXPECT_EQ(ids.count(), 1u);
    EXPECT_TRUE(ids.is_used(5));
    EXPECT_FALSE(ids.is_used(0));

    // Re-acquiring same id is a soft no-op.
    EXPECT_FALSE(ids.acquire(static_cast<uint32_t>(5)));
    EXPECT_EQ(ids.count(), 1u);

    // Filling a gap from acquire() picks the lowest unused via the pool path
    // (the gap ids 0..4 weren't released, just never acquired — so acquire()
    // extends past size() instead). We don't promise gap-filling for ids
    // acquired via specific-id; document by behavior.
    EXPECT_EQ(ids.acquire(), 6u);
}

TEST(FreeList, ReleaseReturnsFalseForFreeOrOutOfRange) {
    FreeList<uint32_t> ids;
    ids.acquire();
    ids.acquire();

    EXPECT_TRUE(ids.release(0));
    EXPECT_FALSE(ids.release(0)); // already free
    EXPECT_FALSE(ids.release(99)); // out of range
}

TEST(FreeList, IteratorSkipsReleasedSlots) {
    FreeList<uint32_t> ids;
    for (int i = 0; i < 5; i++) {
        ids.acquire();
    }
    ids.release(1);
    ids.release(3);

    std::vector<uint32_t> seen;
    for (uint32_t id : ids) {
        seen.push_back(id);
    }
    EXPECT_EQ(seen, (std::vector<uint32_t>{0, 2, 4}));
}

TEST(FreeList, IteratorOnEmptyAndAllReleased) {
    FreeList<uint32_t> ids;
    EXPECT_EQ(ids.begin(), ids.end());

    ids.acquire();
    ids.acquire();
    ids.release(0);
    ids.release(1);
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_EQ(ids.begin(), ids.end());
}
