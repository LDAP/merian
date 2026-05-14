#pragma once

#include "merian/utils/hash.hpp"
#include "merian/utils/small_vector.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <utility>

namespace merian {

// Sorted, unique-element set on top of SmallVector. Suitable as std::map / std::unordered_map key.
template <typename T, std::size_t N, typename Compare = std::less<T>> class SmallSet {
  public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = typename SmallVector<T, N>::iterator;
    using const_iterator = typename SmallVector<T, N>::const_iterator;

    bool empty() const noexcept {
        return data.empty();
    }
    size_type size() const noexcept {
        return data.size();
    }

    iterator begin() noexcept {
        return data.begin();
    }
    iterator end() noexcept {
        return data.end();
    }
    const_iterator begin() const noexcept {
        return data.begin();
    }
    const_iterator end() const noexcept {
        return data.end();
    }

    const T& front() const {
        return data.front();
    }

    std::pair<iterator, bool> insert(const T& v) {
        return insert_impl(v);
    }
    std::pair<iterator, bool> insert(T&& v) {
        return insert_impl(std::move(v));
    }

    size_type erase(const T& v) {
        auto it = std::lower_bound(data.begin(), data.end(), v, comp);
        if (it == data.end() || comp(v, *it))
            return 0;
        data.erase(it);
        return 1;
    }

    bool contains(const T& v) const {
        auto it = std::lower_bound(data.begin(), data.end(), v, comp);
        return it != data.end() && !comp(v, *it);
    }

    // Mirrors std::set::count: returns 0 or 1.
    size_type count(const T& v) const {
        return contains(v) ? 1 : 0;
    }

    void clear() noexcept {
        data.clear();
    }

    bool operator==(const SmallSet& o) const {
        return size() == o.size() && std::equal(begin(), end(), o.begin());
    }
    auto operator<=>(const SmallSet& o) const {
        return std::lexicographical_compare_three_way(begin(), end(), o.begin(), o.end());
    }

  private:
    template <typename U> std::pair<iterator, bool> insert_impl(U&& v) {
        auto it = std::lower_bound(data.begin(), data.end(), v, comp);
        if (it != data.end() && !comp(v, *it))
            return {it, false};
        return {data.insert(it, std::forward<U>(v)), true};
    }

    SmallVector<T, N> data;
    [[no_unique_address]] Compare comp{};
};

} // namespace merian

template <typename T, std::size_t N, typename Compare>
struct std::hash<merian::SmallSet<T, N, Compare>> {
    std::size_t operator()(const merian::SmallSet<T, N, Compare>& s) const noexcept {
        std::size_t seed = 0;
        for (const T& v : s)
            merian::hash_combine(seed, v);
        return seed;
    }
};
