#pragma once

#include "merian/utils/blob.hpp"

#include <cstdint>
#include <filesystem>

namespace merian {

enum class ImageFormat : uint8_t {
    AUTO, // infer from file extension
    PNG,
    JPG,
    BMP,
    TGA,
    HDR,
    PFM,
};

struct ImageInfo {
    int width = 0;
    int height = 0;
    int channels = 0;
};

// Load as 8-bit per channel. Any file format; HDR/PFM source values are tonemapped by the
// loader. desired_channels = 0 keeps the source channel count, 1..4 forces conversion.
// Throws std::runtime_error on failure.
BlobHandle image_load_u8(const std::filesystem::path& path,
                         ImageInfo& info,
                         int desired_channels = 4);

// Load as 32-bit float per channel. Any file format; LDR source values are mapped to [0, 1].
// Throws std::runtime_error on failure.
BlobHandle image_load_f32(const std::filesystem::path& path,
                          ImageInfo& info,
                          int desired_channels = 4);

// Write 8-bit-per-channel data. format=AUTO infers from path. Must be PNG/JPG/BMP/TGA.
// Throws std::runtime_error on failure.
void image_save_u8(const std::filesystem::path& path,
                   const uint8_t* data,
                   int width,
                   int height,
                   int channels,
                   ImageFormat format = ImageFormat::AUTO);

// Write 32-bit-float-per-channel data. format=AUTO infers from path. Must be HDR/PFM.
// Throws std::runtime_error on failure.
void image_save_f32(const std::filesystem::path& path,
                    const float* data,
                    int width,
                    int height,
                    int channels,
                    ImageFormat format = ImageFormat::AUTO);

// Infer ImageFormat from a file extension (case-insensitive). Returns AUTO for unknown.
ImageFormat image_format_from_extension(const std::filesystem::path& path) noexcept;

} // namespace merian
