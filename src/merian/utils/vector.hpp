#pragma once

#include <algorithm>
#include <fmt/format.h>
#include <vector>

namespace merian {

template <class T> void insert_all(std::vector<T>& to, std::vector<T> const& from) {
    std::copy(std::begin(from), std::end(from), std::back_inserter(to));
}

// Copies the memory from `from` to the back of `to`. `to` is accordingly resized.
template <class T> void raw_copy_back(std::vector<T>& to, std::vector<T> const& from) {
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

} // namespace merian
