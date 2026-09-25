#include "merian/vk/utils/image_export.hpp"

#include "merian/utils/defer.hpp"
#include "merian/vk/utils/blits.hpp"

#include <spdlog/spdlog.h>

namespace merian {

const std::vector<std::string>& image_export_format_names() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> upper;
        for (const ImageFormat format : IMAGE_EXPORT_FORMATS) {
            std::string name = image_format_extension(format) + 1; // skip the dot
            for (char& c : name) {
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            }
            upper.push_back(name);
        }
        return upper;
    }();
    return names;
}

void image_export(const ResourceAllocatorHandle& allocator,
                  Submission& submission,
                  const ProfilerHandle& profiler,
                  const ImageHandle& src,
                  const vk::Extent2D& extent,
                  const std::filesystem::path& path,
                  ImageFormat format,
                  ImageMetadata metadata,
                  const bool opaque_alpha) {
    if (format == ImageFormat::AUTO) {
        format = image_format_from_extension(path);
    }
    if (format == ImageFormat::AUTO) {
        throw std::runtime_error{"image_export: cannot infer a format from " + path.string()};
    }
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error && !std::filesystem::is_directory(path.parent_path())) {
        throw std::runtime_error{"image_export: cannot create " + path.parent_path().string() +
                                 ": " + error.message()};
    }
    const bool as_float = format == ImageFormat::HDR || format == ImageFormat::PFM;
    const vk::Extent3D extent_3d{extent.width, extent.height, 1};
    const CommandBufferHandle& cmd = submission.get_cmd();

    const vk::Format capture_format =
        as_float ? vk::Format::eR32G32B32A32Sfloat : vk::Format::eR8G8B8A8Srgb;

    const vk::ImageCreateInfo intermediate_info{
        {},
        vk::ImageType::e2D,
        capture_format,
        extent_3d,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };
    const ImageHandle intermediate_image = allocator->create_image(intermediate_info);
    const BufferHandle staging_buffer = allocator->create_buffer(
        Image::format_size(capture_format) * extent.width * extent.height,
        vk::BufferUsageFlagBits::eTransferDst, MemoryMappingType::HOST_ACCESS_RANDOM);

    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "blit to intermediate image");
        cmd->barrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                     intermediate_image->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                                                 vk::AccessFlagBits::eTransferWrite));
        cmd_blit_stretch(cmd, src, src->get_current_layout(), src->get_extent(), intermediate_image,
                         vk::ImageLayout::eTransferDstOptimal, intermediate_image->get_extent());
        cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                     intermediate_image->barrier(vk::ImageLayout::eTransferSrcOptimal,
                                                 vk::AccessFlagBits::eTransferWrite,
                                                 vk::AccessFlagBits::eTransferRead));
    }
    {
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "copy to buffer");
        // zero row length / image height: tightly packed
        cmd->copy(intermediate_image, staging_buffer,
                  vk::BufferImageCopy{0, 0, 0, first_layer(), {}, extent_3d});
    }
    cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                 staging_buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                vk::AccessFlagBits::eHostRead));

    // unique per export, so concurrent writers to one path cannot share it
    static std::atomic_uint32_t export_counter{0};
    const std::filesystem::path tmp_path =
        path.parent_path() /
        fmt::format(".interm_{}_{}", export_counter++, path.filename().string());

    submission.sync_to_cpu([staging_buffer, path, tmp_path, format, as_float, extent, opaque_alpha,
                            metadata = std::move(metadata)]() {
        const int w = static_cast<int>(extent.width);
        const int h = static_cast<int>(extent.height);
        defer {
            staging_buffer->get_memory()->unmap();
        };

        // the thread pool discards exceptions
        try {
            if (as_float) {
                image_save_f32(tmp_path, staging_buffer->get_memory()->map_as<float>(), w, h, 4,
                               format, metadata);
            } else {
                uint8_t* const rgba = staging_buffer->get_memory()->map_as<uint8_t>();
                if (opaque_alpha) {
                    for (std::size_t i = 3; i < static_cast<std::size_t>(w) * h * 4; i += 4) {
                        rgba[i] = 255;
                    }
                }
                image_save_u8(tmp_path, rgba, w, h, 4, format, metadata);
            }
            try {
                std::filesystem::rename(tmp_path, path);
            } catch (const std::filesystem::filesystem_error&) {
                SPDLOG_WARN("rename failed! Falling back to copy...");
                std::filesystem::copy(tmp_path, path,
                                      std::filesystem::copy_options::overwrite_existing);
                std::filesystem::remove(tmp_path);
            }
        } catch (const std::exception& e) {
            SPDLOG_ERROR("could not write {}: {}", path.string(), e.what());
            return;
        }
        SPDLOG_INFO("wrote image to {}", path.string());
    });
}

TextureHandle image_load_texture(const ResourceAllocatorHandle& allocator,
                                 Submission& submission,
                                 const std::filesystem::path& path,
                                 const vk::ImageUsageFlags additional_usage_flags) {
    ImageInfo info{};
    BlobHandle blob;
    try {
        blob = image_load_f32(path, info, 4);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("{}", e.what());
        return nullptr;
    }
    const TextureHandle texture = allocator->create_texture_from_rgba32f(
        submission.get_cmd(), blob->get_data<float>(), static_cast<uint32_t>(info.width),
        static_cast<uint32_t>(info.height), vk::SamplerAddressMode::eClampToEdge,
        vk::Filter::eLinear, vk::Filter::eLinear, path.filename().string(), false,
        additional_usage_flags);
    submission.get_cmd()->barrier(
        texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
    SPDLOG_INFO("loaded {} ({}x{})", path.string(), info.width, info.height);
    return texture;
}

} // namespace merian
