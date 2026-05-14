#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace merian {

// Vector with inline storage for up to N elements; spills to std::vector beyond N.
template <typename T, std::size_t N> class SmallVector {
    static_assert(N >= 1);
    static_assert(std::is_default_constructible_v<T>);

  public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    bool empty() const noexcept {
        return count == 0;
    }
    size_type size() const noexcept {
        return count;
    }

    T* data() noexcept {
        return is_inline() ? inline_storage.data() : heap_storage.data();
    }
    const T* data() const noexcept {
        return is_inline() ? inline_storage.data() : heap_storage.data();
    }

    iterator begin() noexcept {
        return data();
    }
    iterator end() noexcept {
        return data() + count;
    }
    const_iterator begin() const noexcept {
        return data();
    }
    const_iterator end() const noexcept {
        return data() + count;
    }

    T& front() {
        assert(count > 0);
        return *data();
    }
    const T& front() const {
        assert(count > 0);
        return *data();
    }

    T& operator[](size_type i) {
        return data()[i];
    }
    const T& operator[](size_type i) const {
        return data()[i];
    }

    SmallVector& operator=(std::initializer_list<T> list) {
        clear();
        for (const T& v : list)
            push_back(v);
        return *this;
    }

    void push_back(const T& v) {
        insert(end(), v);
    }
    void push_back(T&& v) {
        insert(end(), std::move(v));
    }

    iterator insert(const_iterator pos, const T& v) {
        return insert_impl(pos, v);
    }
    iterator insert(const_iterator pos, T&& v) {
        return insert_impl(pos, std::move(v));
    }

    iterator erase(const_iterator pos) {
        const auto idx = static_cast<size_type>(pos - data());
        if (is_inline()) {
            std::move(inline_storage.begin() + idx + 1, inline_storage.begin() + count,
                      inline_storage.begin() + idx);
        } else {
            heap_storage.erase(heap_storage.begin() + static_cast<std::ptrdiff_t>(idx));
        }
        --count;
        return data() + idx;
    }

    void clear() noexcept {
        count = 0;
        heap_storage.clear();
    }

  private:
    // Storage stays on the heap once we've spilled, even after erase brings count back
    // below N — otherwise data() starts pointing at stale inline_storage while live
    // elements still live in heap_storage.
    bool is_inline() const noexcept {
        return heap_storage.empty();
    }

    template <typename U> iterator insert_impl(const_iterator pos, U&& v) {
        const auto idx = static_cast<size_type>(pos - data());
        if (!is_inline()) {
            heap_storage.insert(heap_storage.begin() + static_cast<std::ptrdiff_t>(idx),
                                std::forward<U>(v));
        } else if (count < N) {
            std::move_backward(inline_storage.begin() + idx, inline_storage.begin() + count,
                               inline_storage.begin() + count + 1);
            inline_storage[idx] = std::forward<U>(v);
        } else {
            // spill at count == N
            heap_storage.reserve(N + 1);
            heap_storage.insert(heap_storage.end(), std::make_move_iterator(inline_storage.begin()),
                                std::make_move_iterator(inline_storage.end()));
            heap_storage.insert(heap_storage.begin() + static_cast<std::ptrdiff_t>(idx),
                                std::forward<U>(v));
        }
        ++count;
        return data() + idx;
    }

    std::array<T, N> inline_storage{};
    std::vector<T> heap_storage;
    size_type count = 0;
};

} // namespace merian
