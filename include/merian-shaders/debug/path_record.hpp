#pragma once

#include "merian-shaders/debug/path_record.slangh"

#include <algorithm>
#include <array>
#include <cstdint>

namespace merian {

constexpr std::array<const char*, PATH_RECORD_METHOD_COUNT> PATH_RECORD_METHOD_NAMES = {
    "unknown", "uniform", "cosine", "BSDF", "guiding", "ReSTIR", "light"};

inline const char* path_record_method_name(const uint32_t method) {
    return PATH_RECORD_METHOD_NAMES.at(method < PATH_RECORD_METHOD_NAMES.size() ? method : 0);
}

struct PathRecordCapacities {
    uint32_t path_capacity;
    uint32_t vertex_capacity;
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
    };
}

// Smallest buffer whose fixed region split holds the path slots and their vertex blocks.
inline uint64_t path_record_buffer_size(const uint64_t paths, const uint32_t vertices_per_path) {
    const uint64_t vertex_uints = paths * vertices_per_path * PATH_RECORD_VERTEX_UINTS;
    const uint64_t for_vertices =
        vertex_uints * PATH_RECORD_REGION_RATIO / (PATH_RECORD_REGION_RATIO - 1);
    const uint64_t for_paths = paths * PATH_RECORD_REGION_RATIO * PATH_RECORD_UINTS;
    const uint64_t payload = std::max(for_vertices, for_paths) +
                             (static_cast<uint64_t>(PATH_RECORD_REGION_RATIO) * PATH_RECORD_UINTS);
    return (PATH_RECORD_HEADER_UINTS + payload) * 4;
}

} // namespace merian
