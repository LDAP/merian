#pragma once

#include <algorithm>
#include <fmt/format.h>
#include <vector>

namespace merian {

template <class T> void insert_all(std::vector<T>& to, std::vector<T> const& from) {
    to.insert(to.end(), from.begin(), from.end());
}

template <class T>
void insert_range(std::vector<T>& to,
                  std::vector<T> const& from,
                  const std::size_t first,
                  const std::size_t count) {
    assert(first + count < from.size());
    to.insert(to.end(), from.begin() + first, from.begin() + count);
}

// Node: From ends in an indeterminate state
template <class T> void move_all(std::vector<T>& to, const std::vector<T>& from) {
    to.insert(to.end(), std::make_move_iterator(from.begin()), std::make_move_iterator(from.end()));
}

// Node: The range ends in in an indeterminate state (call erase_range)
template <class T>
void move_range(std::vector<T>& to,
                const std::vector<T>& from,
                const std::size_t first,
                const std::size_t count) {
    assert(first + count < from.size());
    to.insert(to.end(), std::make_move_iterator(from.begin() + first),
              std::make_move_iterator(from.begin() + count));
}

template <class T>
void erase_range(std::vector<T>& from, const std::size_t first, const std::size_t count) {
    from.erase(from.begin() + first, from.begin() + count);
}

// Copies the memory from `from` to the back of `to`. `to` is accordingly resized.
template <class T>
    requires std::is_trivially_copyable_v<T>
void raw_copy_back(std::vector<T>& to, std::vector<T> const& from) {
    if (from.empty())
        return;

    std::size_t old_size = to.size();
    to.resize(old_size + from.size());
    memcpy(&to[old_size], from.data(), from.size() * sizeof(T));
}

template <class T> void check_size(const std::vector<T>& vector, std::size_t index) {
    if (index >= vector.size()) {
        throw std::runtime_error(fmt::format("Index {} invalid for size {}", index, vector.size()));
    }
}

template <class T> void remove_duplicates(std::vector<T>& vector) {
    std::sort(vector.begin(), vector.end());
    vector.erase(std::unique(vector.begin(), vector.end()), vector.end());
}

// Returns the size of the vector in bytes.
template <class T> inline std::size_t size_of(const std::vector<T>& vector) {
    return vector.size() * sizeof(T);
}

} // namespace merian
