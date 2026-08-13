#pragma once

#include <array>
#include <cstdint>

namespace merian {

// Names of PATH_RECORD_METHOD_* in path-record.slang; index 0 covers every id this build does
// not know about.
constexpr std::array<const char*, 7> PATH_RECORD_METHOD_NAMES = {
    "unknown", "uniform", "cosine", "BSDF", "MCPG", "ReSTIR", "SSMM"};

inline const char* path_record_method_name(const uint32_t method) {
    return PATH_RECORD_METHOD_NAMES.at(method < PATH_RECORD_METHOD_NAMES.size() ? method : 0);
}

// Host mirror of the record stream layout in debug/path-record.slang.
constexpr uint32_t PATH_RECORD_HEADER_UINTS = 16;
// paths get 1 of PATH_RECORD_REGION_RATIO payload parts, vertices the rest
constexpr uint32_t PATH_RECORD_REGION_RATIO = 17;
constexpr uint32_t PATH_RECORD_UINTS = 4;
constexpr uint32_t PATH_RECORD_VERTEX_UINTS = 8;
constexpr uint32_t PATH_RECORD_LOBE_SHIFT = 8;
constexpr uint32_t PATH_RECORD_LOBE_MASK = 0xFu;
constexpr uint32_t PATH_RECORD_FLAG_TERMINAL = 1u << 12;
constexpr uint32_t PATH_RECORD_FLAG_NEE = 1u << 13;
constexpr uint32_t PATH_RECORD_METHOD_SHIFT = 16;
constexpr uint32_t PATH_RECORD_METHOD_MASK = 0xFu;
constexpr uint32_t PATH_RECORD_MATERIAL_SHIFT = 20;
constexpr uint32_t PATH_RECORD_MATERIAL_MASK = 0xFFFu;
constexpr uint32_t PATH_RECORD_MATERIAL_NONE = 0xFFFu;
constexpr uint32_t PATH_RECORD_ALL_PIXELS = 0xFFFFFFFFu;

struct PathRecordCapacities {
    uint32_t path_capacity;
    uint32_t vertex_capacity;
    uint32_t vertex_region_offset; // in uints
};

// Must match path_record_path_capacity() / path_record_vertex_capacity() in path-record.slang.
inline PathRecordCapacities path_record_capacities(const uint64_t buffer_bytes) {
    const uint64_t payload = (buffer_bytes / 4) - PATH_RECORD_HEADER_UINTS;
    const auto path_capacity =
        static_cast<uint32_t>(payload / PATH_RECORD_REGION_RATIO / PATH_RECORD_UINTS);
    const uint64_t path_uints = static_cast<uint64_t>(path_capacity) * PATH_RECORD_UINTS;
    return PathRecordCapacities{
        .path_capacity = path_capacity,
        .vertex_capacity = static_cast<uint32_t>((payload - path_uints) / PATH_RECORD_VERTEX_UINTS),
        .vertex_region_offset = PATH_RECORD_HEADER_UINTS + (path_capacity * PATH_RECORD_UINTS),
    };
}

// Smallest buffer whose fixed region split fits the given number of full vertex blocks.
inline uint64_t path_record_buffer_size(const uint64_t paths, const uint32_t vertices_per_path) {
    const uint64_t vertex_uints = paths * vertices_per_path * PATH_RECORD_VERTEX_UINTS;
    const uint64_t payload = (vertex_uints * PATH_RECORD_REGION_RATIO / 16) +
                             (static_cast<uint64_t>(PATH_RECORD_REGION_RATIO) * PATH_RECORD_UINTS);
    return (PATH_RECORD_HEADER_UINTS + payload) * 4;
}

} // namespace merian
