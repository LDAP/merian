#pragma once

#include "merian/io/image_io.hpp"
#include "merian/vk/command/submission.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace merian {

// Format choice offered by the nodes that write images, in the order their combo lists them.
constexpr std::array<ImageFormat, 4> IMAGE_EXPORT_FORMATS = {ImageFormat::PNG, ImageFormat::JPG,
                                                             ImageFormat::HDR, ImageFormat::PFM};

// Labels for the format combo, aligned with IMAGE_EXPORT_FORMATS.
const std::vector<std::string>& image_export_format_names();

// Blits src (must be in a transfer-src compatible layout) into a tightly-packed staging buffer
// inside the submission, then encodes and writes the file once the GPU work completed, through a
// temp file so watchers never see a partial one. Float formats capture linear f32, the rest sRGB
// u8; extent may differ from the src extent to rescale.
//
// Throws std::runtime_error for anything decidable up front (unknown format, unusable
// destination). An I/O failure after the GPU round-trip is logged instead.
void image_export(const ResourceAllocatorHandle& allocator,
                  Submission& submission,
                  const ProfilerHandle& profiler,
                  const ImageHandle& src,
                  const vk::Extent2D& extent,
                  const std::filesystem::path& path,
                  ImageFormat format = ImageFormat::AUTO,
                  ImageMetadata metadata = {});

} // namespace merian
