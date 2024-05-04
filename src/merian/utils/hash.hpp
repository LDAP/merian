#pragma once
#include <cstdint>
#include <functional>

namespace merian {

//---- Hash Combination ----
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3876.pdf
template <typename T> void hash_combine(std::size_t& seed, const T& val) {
    seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
// Auxiliary generic functions to create a hash value using a seed
template <typename T, typename... Types> void hash_combine(std::size_t& seed, const T& val, const Types&... args) {
    hash_combine(seed, val);
    hash_combine(seed, args...);
}
// Optional auxiliary generic functions to support hash_val() without arguments
inline void hashCombine(std::size_t&) {}

// Generic function to create a hash value out of a heterogeneous list of arguments
template <typename... Types> std::size_t hash_val(const Types&... args) {
    std::size_t seed = 0;
    hash_combine(seed, args...);
    return seed;
}
//--------------

template <typename T> std::size_t hash_aligned_32(const T& v) {
    const uint32_t size = sizeof(T) / sizeof(uint32_t);
    const uint32_t* vBits = reinterpret_cast<const uint32_t*>(&v);
    std::size_t seed = 0;
    for (uint32_t i = 0u; i < size; i++) {
        hash_combine(seed, vBits[i]);
    }
    return seed;
}

// Generic hash function to use when using a struct aligned to 32-bit as std::map-like container key
// Important: this only works if the struct contains integral types, as it will not
// do any pointer chasing
template <typename T> struct HashAligned32 {
    std::size_t operator()(const T& s) const {
        return hash_aligned_32(s);
    }
};

} // namespace merian
