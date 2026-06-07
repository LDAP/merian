#pragma once

#include "merian/io/image_io.hpp"
#include "merian/utils/blob.hpp"

#include <vulkan/vulkan.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace merian {

// A parsed BCn-compressed DDS image. `data` holds the compressed blocks of all mip levels,
// concatenated mip 0 first.
struct DdsImage {
    vk::Format format = vk::Format::eUndefined;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mip_levels = 1;
    // True if the format carries an alpha channel.
    bool has_alpha = false;
    std::vector<uint8_t> data;
};

// True if the path has a .dds extension (case-insensitive).
bool is_dds(const std::filesystem::path& path);

// Parse a BCn-compressed DDS file. `srgb` selects the sRGB vs UNORM variant for color formats.
// Throws on failure or an unsupported format.
DdsImage dds_load(const std::filesystem::path& path, bool srgb);

// CPU-decode mip 0 of a BCn DDS image to RGBA8 (4 channels). Supports BC1-BC5; throws for BC6H/BC7.
BlobHandle dds_decode_rgba8(const DdsImage& dds, ImageInfo& info);

} // namespace merian
