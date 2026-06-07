#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/io/dds.hpp"
#include "merian/io/image_io.hpp"
#include "merian/utils/colors.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/utils/blits.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <spdlog/spdlog.h>

namespace merian {

ResourceAllocator::ResourceAllocator(const ContextHandle& context,
                                     const std::shared_ptr<MemoryAllocator>& memAllocator,
                                     const std::shared_ptr<StagingMemoryManager>& staging,
                                     const std::shared_ptr<SamplerPool>& samplerPool,
                                     const DescriptorSetAllocatorHandle& descriptor_pool)
    : context(context), m_memAlloc(memAllocator), m_staging(staging), m_samplerPool(samplerPool),
      descriptor_pool(descriptor_pool),
      debug_utils(context->get_context_extension<ExtensionVkDebugUtils>(true)) {
    SPDLOG_DEBUG("create ResourceAllocator ({})", fmt::ptr(this));

    const uint32_t missing_rgba = merian::uint32_from_rgba(1, 0, 1, 1);
    const std::vector<uint32_t> data = {missing_rgba, missing_rgba, missing_rgba, missing_rgba};
    context->get_queue_GCT()->submit_wait([&](const CommandBufferHandle& cmd) {
        const ImageHandle dummy_storage_image =
            create_image_from_rgba8(cmd, data.data(), 2, 2, vk::ImageUsageFlagBits::eStorage, false,
                                    1, "ResourceAllocator::dummy_storage_image");
        dummy_storage_image_view = ImageView::create(dummy_storage_image);

        const auto img_transition = dummy_storage_image->barrier2(vk::ImageLayout::eGeneral);
        dummy_texture = create_texture_from_rgba8(
            cmd, data.data(), 2, 2, vk::SamplerAddressMode::eRepeat, vk::Filter::eNearest,
            vk::Filter::eNearest, true, "ResourceAllocator::dummy_texture");
        const auto tex_transition =
            dummy_texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal);

        cmd->barrier({img_transition, tex_transition});

        dummy_buffer = create_buffer(cmd, data.size() * sizeof(uint32_t),
                                     vk::BufferUsageFlagBits::eStorageBuffer, data.data(),
                                     MemoryMappingType::NONE, "ResourceAllocator::dummy_buffer");
    });

    SPDLOG_DEBUG("Uploaded dummy texture and buffer");
}

BufferHandle ResourceAllocator::create_buffer(const vk::BufferCreateInfo& info,
                                              const MemoryMappingType mapping_type,
                                              const std::string& debug_name,
                                              const std::optional<vk::DeviceSize> min_alignment) {
    const BufferHandle buffer =
        m_memAlloc->create_buffer(info, mapping_type, debug_name, min_alignment);

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->get_device()->get_device(), **buffer, debug_name);
    }
    SPDLOG_TRACE("created buffer {} ({})", fmt::ptr(static_cast<VkBuffer>(**buffer)), debug_name);
#endif

    return buffer;
}

BufferHandle ResourceAllocator::create_buffer(const vk::DeviceSize size_,
                                              const vk::BufferUsageFlags usage_,
                                              const MemoryMappingType mapping_type,
                                              const std::string& debug_name,
                                              const std::optional<vk::DeviceSize> min_alignment) {
    vk::BufferCreateInfo info{{}, size_, vk::BufferUsageFlagBits::eTransferDst | usage_};
    return create_buffer(info, mapping_type, debug_name, min_alignment);
}

BufferHandle ResourceAllocator::create_buffer(const CommandBufferHandle& cmdBuf,
                                              const vk::DeviceSize& size_,
                                              const vk::BufferUsageFlags usage_,
                                              const void* data_,
                                              const MemoryMappingType mapping_type,
                                              const std::string& debug_name,
                                              const std::optional<vk::DeviceSize> min_alignment) {
    BufferHandle resultBuffer =
        create_buffer(size_, usage_, mapping_type, debug_name, min_alignment);

    if (data_ != nullptr) {
        m_staging->cmd_to_device(cmdBuf, resultBuffer, data_);
    }

    return resultBuffer;
}

const BufferHandle& ResourceAllocator::get_dummy_buffer() const {
    return dummy_buffer;
}

bool ResourceAllocator::ensure_buffer_size(BufferHandle& buffer,
                                           const vk::DeviceSize buffer_size,
                                           const vk::BufferUsageFlags usage,
                                           const std::string& debug_name,
                                           const MemoryMappingType mapping_type,
                                           const std::optional<vk::DeviceSize> min_alignment,
                                           const float growth_factor) {
    assert(growth_factor >= 1);

    if (buffer && buffer->get_size() >= buffer_size) {
        return false;
    }

    buffer =
        create_buffer(buffer_size * growth_factor, usage, mapping_type, debug_name, min_alignment);
    return true;
}

// You get the alignment from
// VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment
BufferHandle ResourceAllocator::create_scratch_buffer(const vk::DeviceSize size,
                                                      const vk::DeviceSize alignment,
                                                      const std::string& debug_name) {
    return create_buffer(size, Buffer::SCRATCH_BUFFER_USAGE, MemoryMappingType::NONE, debug_name,
                         alignment);
}

BufferHandle ResourceAllocator::create_instances_buffer(const uint32_t instance_count,
                                                        const std::string& debug_name) {

    return create_buffer(sizeof(vk::AccelerationStructureInstanceKHR) * instance_count,
                         Buffer::INSTANCES_BUFFER_USAGE, merian::MemoryMappingType::NONE,
                         debug_name, 16);
}

ImageHandle ResourceAllocator::create_image(const vk::ImageCreateInfo& info_,
                                            const MemoryMappingType mapping_type,
                                            const std::string& debug_name) {
    const ImageHandle image = m_memAlloc->create_image(info_, mapping_type, debug_name);

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->get_device()->get_device(), **image, debug_name);
    }
    SPDLOG_TRACE("created image {} ({})", fmt::ptr(static_cast<VkImage>(**image)), debug_name);
#endif

    return image;
}

ImageHandle ResourceAllocator::create_image(const CommandBufferHandle& cmdBuf,
                                            const void* data_,
                                            const vk::ImageCreateInfo& info_,
                                            const MemoryMappingType mapping_type,
                                            const std::string& debug_name) {
    assert(data_);

    const ImageHandle result_image = create_image(info_, mapping_type, debug_name);

    // doing these transitions per copy is not efficient, should do in bulk for many images
    cmdBuf->barrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                    result_image->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                                          vk::AccessFlagBits::eTransferWrite,
                                          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                          all_levels_and_layers(), true));

    m_staging->cmd_to_device(cmdBuf, result_image, data_);

    return result_image;
}

ImageHandle ResourceAllocator::create_image_from_rgba8(const CommandBufferHandle& cmd,
                                                       const uint32_t* data,
                                                       const uint32_t width,
                                                       const uint32_t height,
                                                       const vk::ImageUsageFlags usage,
                                                       const bool isSRGB,
                                                       const uint32_t mip_levels,
                                                       const std::string& debug_name) {
    const vk::ImageCreateInfo tex_image_info{
        {},
        vk::ImageType::e2D,
        isSRGB ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
        {width, height, 1},
        mip_levels,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        usage | vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };

    // transfers all levels to TransferDstOptimal
    return create_image(cmd, data, tex_image_info, MemoryMappingType::NONE, debug_name);
}

ImageHandle ResourceAllocator::create_image_from_rgba32f(const CommandBufferHandle& cmd,
                                                         const float* data,
                                                         const uint32_t width,
                                                         const uint32_t height,
                                                         const vk::ImageUsageFlags usage,
                                                         const uint32_t mip_levels,
                                                         const std::string& debug_name) {
    const vk::ImageCreateInfo tex_image_info{
        {},
        vk::ImageType::e2D,
        vk::Format::eR32G32B32A32Sfloat,
        {width, height, 1},
        mip_levels,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        usage | vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };

    return create_image(cmd, data, tex_image_info, MemoryMappingType::NONE, debug_name);
}

const ImageViewHandle& ResourceAllocator::get_dummy_storage_image_view() const {
    return dummy_storage_image_view;
}

ImageViewHandle
ResourceAllocator::create_image_view(const ImageHandle& image,
                                     const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                     [[maybe_unused]] const std::string& debug_name) {

    const ImageViewHandle view = ImageView::create(imageViewCreateInfo, image);

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->get_device()->get_device(), **view, debug_name);
    }
    SPDLOG_TRACE("created image view {} ({}), for image {}",
                 fmt::ptr(static_cast<VkImageView>(**view)), debug_name,
                 fmt::ptr(static_cast<VkImage>(**image)));
#endif

    return view;
}

TextureHandle ResourceAllocator::create_texture(const ImageHandle& image,
                                                const vk::ImageViewCreateInfo& view_create_info,
                                                const SamplerHandle& sampler,
                                                [[maybe_unused]] const std::string& debug_name) {
    const ImageViewHandle view = create_image_view(image, view_create_info, debug_name);

    return Texture::create(view, sampler);
}

TextureHandle ResourceAllocator::create_texture(const ImageHandle& image,
                                                const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                                const vk::SamplerCreateInfo& samplerCreateInfo,
                                                const std::string& debug_name) {
    const SamplerHandle sampler = m_samplerPool->acquire_sampler(samplerCreateInfo);
    return create_texture(image, imageViewCreateInfo, sampler, debug_name);
}

TextureHandle ResourceAllocator::create_texture(const ImageHandle& image,
                                                const std::string& debug_name) {
    return create_texture(image, image->make_view_create_info(), debug_name);
}

TextureHandle ResourceAllocator::create_texture(const ImageHandle& image,
                                                const vk::ImageViewCreateInfo& view_create_info,
                                                const std::string& debug_name) {
    const ContextHandle& context = image->get_memory()->get_context();

    const vk::FormatProperties props =
        context->get_physical_device()->get_physical_device().getFormatProperties(
            view_create_info.format);

    SamplerHandle sampler;
    if ((image->get_tiling() == vk::ImageTiling::eOptimal &&
         (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) ||
        (image->get_tiling() == vk::ImageTiling::eLinear &&
         (props.linearTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))) {
        sampler = get_sampler_pool()->linear_mirrored_repeat();
    } else {
        sampler = get_sampler_pool()->nearest_mirrored_repeat();
    }

    return create_texture(image, view_create_info, sampler, debug_name);
}

TextureHandle
ResourceAllocator::create_texture_from_rgba8(const CommandBufferHandle& cmd,
                                             const uint32_t* data,
                                             const uint32_t width,
                                             const uint32_t height,
                                             const SamplerHandle& sampler,
                                             const bool isSRGB,
                                             const std::string& debug_name,
                                             const bool generate_mipmaps,
                                             const vk::ImageUsageFlags additional_usage_flags) {
    uint32_t mip_levels = 1;
    vk::ImageUsageFlags usage_flags = vk::ImageUsageFlagBits::eSampled | additional_usage_flags;
    if (generate_mipmaps) {
        mip_levels = static_cast<uint32_t>(floor(log2(std::max(width, height))) + 1);
        usage_flags |= vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    }

    const merian::ImageHandle image = create_image_from_rgba8(cmd, data, width, height, usage_flags,
                                                              isSRGB, mip_levels, debug_name);

    cmd_generate_mipmaps(cmd, image);

    return create_texture(image, image->make_view_create_info(), sampler, debug_name);
}

TextureHandle
ResourceAllocator::create_texture_from_rgba8(const CommandBufferHandle& cmd,
                                             const uint32_t* data,
                                             const uint32_t width,
                                             const uint32_t height,
                                             const vk::SamplerAddressMode address_mode,
                                             const vk::Filter mag_filter,
                                             const vk::Filter min_filter,
                                             const bool isSRGB,
                                             const std::string& debug_name,
                                             const bool generate_mipmaps,
                                             const vk::ImageUsageFlags additional_usage_flags) {
    const SamplerHandle sampler =
        get_sampler_pool()->for_filter_and_address_mode(mag_filter, min_filter, address_mode);
    return create_texture_from_rgba8(cmd, data, width, height, sampler, isSRGB, debug_name,
                                     generate_mipmaps, additional_usage_flags);
}

TextureHandle
ResourceAllocator::create_texture_from_rgba32f(const CommandBufferHandle& cmd,
                                               const float* data,
                                               const uint32_t width,
                                               const uint32_t height,
                                               const vk::SamplerAddressMode address_mode,
                                               const vk::Filter mag_filter,
                                               const vk::Filter min_filter,
                                               const std::string& debug_name,
                                               const bool generate_mipmaps,
                                               const vk::ImageUsageFlags additional_usage_flags) {
    uint32_t mip_levels = 1;
    vk::ImageUsageFlags usage_flags = vk::ImageUsageFlagBits::eSampled | additional_usage_flags;
    if (generate_mipmaps) {
        mip_levels = static_cast<uint32_t>(floor(log2(std::max(width, height))) + 1);
        usage_flags |= vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    }

    const ImageHandle image =
        create_image_from_rgba32f(cmd, data, width, height, usage_flags, mip_levels, debug_name);

    cmd_generate_mipmaps(cmd, image);

    const SamplerHandle sampler =
        get_sampler_pool()->for_filter_and_address_mode(mag_filter, min_filter, address_mode);
    return create_texture(image, image->make_view_create_info(), sampler, debug_name);
}

static uint32_t compressed_block_size(const vk::Format format) {
    switch (format) {
    case vk::Format::eBc1RgbUnormBlock:
    case vk::Format::eBc1RgbSrgbBlock:
    case vk::Format::eBc1RgbaUnormBlock:
    case vk::Format::eBc1RgbaSrgbBlock:
    case vk::Format::eBc4UnormBlock:
    case vk::Format::eBc4SnormBlock:
        return 8;
    default:
        return 16; // BC2/BC3/BC5/BC6H/BC7
    }
}

TextureHandle ResourceAllocator::create_texture_from_compressed(const CommandBufferHandle& cmd,
                                                                const vk::Format format,
                                                                const uint32_t width,
                                                                const uint32_t height,
                                                                const uint32_t mip_levels,
                                                                const void* data,
                                                                const vk::DeviceSize data_size,
                                                                const SamplerHandle& sampler,
                                                                const std::string& debug_name) {
    const vk::ImageCreateInfo image_info{
        {},
        vk::ImageType::e2D,
        format,
        {width, height, 1},
        mip_levels,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };
    const ImageHandle image = create_image(image_info, MemoryMappingType::NONE, debug_name);

    const BufferHandle staging =
        create_buffer(data_size, vk::BufferUsageFlagBits::eTransferSrc,
                      MemoryMappingType::HOST_ACCESS_SEQUENTIAL_WRITE, "compressed staging");
    std::memcpy(staging->get_memory()->map(), data, data_size);
    staging->get_memory()->unmap();

    const uint32_t block_bytes = compressed_block_size(format);
    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(mip_levels);
    vk::DeviceSize offset = 0;
    uint32_t mw = width;
    uint32_t mh = height;
    for (uint32_t mip = 0; mip < mip_levels; mip++) {
        const vk::DeviceSize size = static_cast<vk::DeviceSize>(std::max(1u, (mw + 3) / 4)) *
                                    std::max(1u, (mh + 3) / 4) * block_bytes;
        if (offset + size > data_size) {
            break; // truncated data; stop at the last fully-present mip
        }
        regions.push_back(vk::BufferImageCopy{
            offset, 0, 0, {vk::ImageAspectFlagBits::eColor, mip, 0, 1}, {0, 0, 0}, {mw, mh, 1}});
        offset += size;
        mw = std::max(1u, mw / 2);
        mh = std::max(1u, mh / 2);
    }

    cmd->barrier(image->barrier2(vk::ImageLayout::eTransferDstOptimal));
    cmd->get_command_buffer().copyBufferToImage(staging->get_buffer(), **image,
                                                vk::ImageLayout::eTransferDstOptimal, regions);
    cmd->barrier(image->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
    cmd->keep_until_pool_reset(staging);

    return create_texture(image, image->make_view_create_info(), sampler, debug_name);
}

TextureHandle ResourceAllocator::create_texture_from_file(const CommandBufferHandle& cmd,
                                                          const std::filesystem::path& path,
                                                          const bool srgb,
                                                          const vk::SamplerAddressMode address_mode,
                                                          const vk::Filter mag_filter,
                                                          const vk::Filter min_filter,
                                                          const std::string& debug_name,
                                                          const bool generate_mipmaps,
                                                          bool* out_has_alpha) {
    // BCn DDS: upload the raw compressed blocks (and its stored mip chain) directly.
    if (is_dds(path)) {
        const DdsImage dds = dds_load(path, srgb);
        if (out_has_alpha != nullptr) {
            *out_has_alpha = dds.has_alpha;
        }
        const SamplerHandle sampler = m_samplerPool->for_filter_and_address_mode(
            mag_filter, min_filter, address_mode, vk::SamplerMipmapMode::eLinear);
        return create_texture_from_compressed(cmd, dds.format, dds.width, dds.height,
                                              dds.mip_levels, dds.data.data(), dds.data.size(),
                                              sampler, debug_name);
    }

    // Everything else: decode to RGBA8 via the host-side stb loader.
    ImageInfo info;
    const BlobHandle blob = image_load_u8(path, info, 4);
    if (out_has_alpha != nullptr) {
        *out_has_alpha = info.source_channels == 4;
    }
    const TextureHandle texture = create_texture_from_rgba8(
        cmd, blob->get_data<uint32_t>(), static_cast<uint32_t>(info.width),
        static_cast<uint32_t>(info.height), address_mode, mag_filter, min_filter, srgb, debug_name,
        generate_mipmaps);
    cmd->barrier(texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal));
    return texture;
}

const TextureHandle& ResourceAllocator::get_dummy_texture() const {
    return dummy_texture;
}

AccelerationStructureHandle ResourceAllocator::create_acceleration_structure(
    const vk::AccelerationStructureTypeKHR type,
    const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
    const std::string& debug_name) {
    // Allocating the buffer to hold the acceleration structure
    BufferHandle buffer = create_buffer(size_info.accelerationStructureSize,
                                        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                            vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                        MemoryMappingType::NONE, debug_name);
    vk::AccelerationStructureKHR as;
    // Setting the buffer
    vk::AccelerationStructureCreateInfoKHR create_info{
        {}, *buffer, {}, size_info.accelerationStructureSize, type};
    check_result(context->get_device()->get_device().createAccelerationStructureKHR(&create_info,
                                                                                    nullptr, &as),
                 "could not create acceleration structure");

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->get_device()->get_device(), **buffer,
                                     debug_name + " buffer");
        debug_utils->set_object_name(context->get_device()->get_device(), as, debug_name);
    }
#endif

    return AccelerationStructure::create(as, buffer);
}

DescriptorSetHandle
ResourceAllocator::allocate_descriptor_set(const DescriptorSetLayoutHandle& layout) {
    return descriptor_pool->allocate(layout);
}

std::vector<DescriptorSetHandle>
ResourceAllocator::allocate_descriptor_set(const DescriptorSetLayoutHandle& layout,
                                           const uint32_t set_count) {
    return descriptor_pool->allocate(layout, set_count);
}

std::shared_ptr<StagingMemoryManager> ResourceAllocator::get_staging() {
    return m_staging;
}

const std::shared_ptr<StagingMemoryManager>& ResourceAllocator::get_staging() const {
    return m_staging;
}

} // namespace merian
