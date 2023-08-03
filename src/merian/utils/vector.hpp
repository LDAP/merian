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

//---- Hash Combination ----
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3876.pdf
template <typename T> void hashCombine(std::size_t& seed, const T& val) {
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
// Auxiliary generic functions to create a hash value using a seed
template <typename T, typename... Types> void hashCombine(std::size_t& seed, const T& val, const Types&... args) {
    hashCombine(seed, val);
    hashCombine(seed, args...);
}
// Optional auxiliary generic functions to support hash_val() without arguments
inline void hashCombine(std::size_t&) {}

// Generic function to create a hash value out of a heterogeneous list of arguments
template <typename... Types> std::size_t hashVal(const Types&... args) {
    std::size_t seed = 0;
    hashCombine(seed, args...);
    return seed;
}
//--------------

template <typename T> std::size_t hashAligned32(const T& v) {
    const uint32_t size = sizeof(T) / sizeof(uint32_t);
    const uint32_t* vBits = reinterpret_cast<const uint32_t*>(&v);
    std::size_t seed = 0;
    for (uint32_t i = 0u; i < size; i++) {
        hashCombine(seed, vBits[i]);
    }
    return seed;
}

// Generic hash function to use when using a struct aligned to 32-bit as std::map-like container key
// Important: this only works if the struct contains integral types, as it will not
// do any pointer chasing
template <typename T> struct HashAligned32 {
    std::size_t operator()(const T& s) const {
        return hashAligned32(s);
    }
};

} // namespace merian
