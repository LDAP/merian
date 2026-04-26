#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

namespace merian {

// Dense-id allocator with tail compaction.
template <typename ID = uint32_t> class FreeList {
  public:
    class Iterator {
      public:
        Iterator(const FreeList* list, ID id) : list(list), id(id) {
            skip_unused();
        }

        ID operator*() const {
            return id;
        }

        Iterator& operator++() {
            ++id;
            skip_unused();
            return *this;
        }

        bool operator==(const Iterator& other) const {
            return id == other.id && list == other.list;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

      private:
        void skip_unused() {
            while (id < list->used.size() && !list->used[id]) {
                ++id;
            }
        }

        const FreeList* list;
        ID id;
    };

    // Claim a new ID. Reuses previously released IDs.
    ID acquire() {
        // skip stale entries left over from tail compaction
        while (!free_pool.empty() && free_pool.back() >= used.size()) {
            free_pool.pop_back();
        }
        if (!free_pool.empty()) {
            const ID id = free_pool.back();
            free_pool.pop_back();
            assert(!used[id]);
            used[id] = true;
            ++used_count;
            return id;
        }
        const ID id = static_cast<ID>(used.size());
        used.push_back(true);
        ++used_count;
        return id;
    }

    // Claim a specific ID. Returns false if the ID is already in use.
    bool acquire(const ID id) {
        if (id < used.size() && used[id]) {
            return false;
        }
        if (id >= used.size()) {
            used.resize(id + 1, false);
        }
        used[id] = true;
        ++used_count;
        return true;
    }

    // Releases an ID. Returns false if the ID was unused.
    bool release(const ID id) {
        if (id >= used.size() || !used[id]) {
            return false;
        }
        used[id] = false;
        --used_count;
        if (id == used.size() - 1) {
            // trim trailing released slots; stale free_pool entries filtered in acquire().
            while (!used.empty() && !used.back()) {
                used.pop_back();
            }
        } else {
            free_pool.push_back(id);
        }
        return true;
    }

    // Number of used IDs.
    uint32_t count() const {
        return used_count;
    }

    // 1 + the largest used id (i.e. capacity needed by vectors storing data for each ID).
    uint32_t size() const {
        return static_cast<uint32_t>(used.size());
    }

    bool is_used(const ID id) const {
        return id < used.size() && used[id];
    }

    Iterator begin() const {
        return Iterator(this, 0);
    }

    Iterator end() const {
        return Iterator(this, static_cast<ID>(used.size()));
    }

  private:
    std::vector<bool> used;
    std::vector<ID> free_pool;
    uint32_t used_count = 0;
};

} // namespace merian
