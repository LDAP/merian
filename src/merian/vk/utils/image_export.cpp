#include "merian/vk/utils/image_export.hpp"

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
                  ImageMetadata metadata) {
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
        // zero row length / image height: rows are tightly packed, unlike a linear image which
        // carries a driver-defined row pitch.
        cmd->copy(intermediate_image, staging_buffer,
                  vk::BufferImageCopy{0, 0, 0, first_layer(), {}, extent_3d});
    }
    cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                 staging_buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite,
                                                vk::AccessFlagBits::eHostRead));

    // unique per export, so concurrent writers to one path cannot share a temp file
    static std::atomic_uint32_t export_counter{0};
    const std::filesystem::path tmp_path =
        path.parent_path() /
        fmt::format(".interm_{}_{}", export_counter++, path.filename().string());

    submission.sync_to_cpu([staging_buffer, path, tmp_path, format, as_float, extent,
                            metadata = std::move(metadata)]() {
        const int w = static_cast<int>(extent.width);
        const int h = static_cast<int>(extent.height);

        // the thread pool discards exceptions, report here or the capture fails silently
        try {
            if (format == ImageFormat::PFM) {
                // PFM has no alpha channel
                const float* rgba = staging_buffer->get_memory()->map_as<float>();
                std::vector<float> rgb(static_cast<std::size_t>(w) * h * 3);
                for (std::size_t i = 0; i < static_cast<std::size_t>(w) * h; i++) {
                    rgb[(i * 3) + 0] = rgba[(i * 4) + 0];
                    rgb[(i * 3) + 1] = rgba[(i * 4) + 1];
                    rgb[(i * 3) + 2] = rgba[(i * 4) + 2];
                }
                image_save_f32(tmp_path, rgb.data(), w, h, 3, format, metadata);
            } else if (as_float) {
                image_save_f32(tmp_path, staging_buffer->get_memory()->map_as<float>(), w, h, 4,
                               format, metadata);
            } else {
                image_save_u8(tmp_path, staging_buffer->get_memory()->map_as<uint8_t>(), w, h, 4,
                              format, metadata);
            }
        } catch (const std::exception& e) {
            SPDLOG_ERROR("could not write {}: {}", path.string(), e.what());
            staging_buffer->get_memory()->unmap();
            return;
        }

        try {
            std::filesystem::rename(tmp_path, path);
        } catch (const std::filesystem::filesystem_error&) {
            SPDLOG_WARN("rename failed! Falling back to copy...");
            std::filesystem::copy(tmp_path, path);
            std::filesystem::remove(tmp_path);
        }
        SPDLOG_INFO("wrote image to {}", path.string());

        staging_buffer->get_memory()->unmap();
    });
}

} // namespace merian
