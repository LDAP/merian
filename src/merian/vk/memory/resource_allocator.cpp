#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/utils/colors.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/extension/extension_vk_debug_utils.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <spdlog/spdlog.h>

namespace merian {

ResourceAllocator::ResourceAllocator(const ContextHandle& context,
                                     const std::shared_ptr<MemoryAllocator>& memAllocator,
                                     const std::shared_ptr<StagingMemoryManager>& staging,
                                     const std::shared_ptr<SamplerPool>& samplerPool)
    : context(context), m_memAlloc(memAllocator), m_staging(staging), m_samplerPool(samplerPool),
      debug_utils(context->get_extension<ExtensionVkDebugUtils>()) {
    SPDLOG_DEBUG("create ResourceAllocator ({})", fmt::ptr(this));

    const uint32_t missing_rgba = merian::uint32_from_rgba(1, 0, 1, 1);
    const std::vector<uint32_t> data = {missing_rgba, missing_rgba, missing_rgba, missing_rgba};
    context->get_queue_GCT()->submit_wait([&](const CommandBufferHandle& cmd) {
        const ImageHandle dummy_storage_image =
            createImageFromRGBA8(cmd, data.data(), 2, 2, vk::ImageUsageFlagBits::eStorage, false, 1,
                                 "ResourceAllocator::dummy_storage_image");
        dummy_storage_image_view = ImageView::create(dummy_storage_image);

        const auto img_transition = dummy_storage_image->barrier2(vk::ImageLayout::eGeneral);
        dummy_texture =
            createTextureFromRGBA8(cmd, data.data(), 2, 2, vk::Filter::eNearest,
                                   vk::Filter::eNearest, true, "ResourceAllocator::dummy_texture");
        const auto tex_transition =
            dummy_texture->get_image()->barrier2(vk::ImageLayout::eShaderReadOnlyOptimal);

        cmd->barrier({img_transition, tex_transition});

        dummy_buffer = createBuffer(cmd, data.size() * sizeof(uint32_t),
                                    vk::BufferUsageFlagBits::eStorageBuffer, data.data(),
                                    MemoryMappingType::NONE, "ResourceAllocator::dummy_buffer");
    });

    SPDLOG_DEBUG("Uploaded dummy texture and buffer");
}

BufferHandle ResourceAllocator::createBuffer(const vk::BufferCreateInfo& info,
                                             const MemoryMappingType mapping_type,
                                             const std::string& debug_name,
                                             const std::optional<vk::DeviceSize> min_alignment) {
    const BufferHandle buffer =
        m_memAlloc->create_buffer(info, mapping_type, debug_name, min_alignment);

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->device, **buffer, debug_name);
    }
    SPDLOG_TRACE("created buffer {} ({})", fmt::ptr(static_cast<VkBuffer>(**buffer)), debug_name);
#endif

    return buffer;
}

BufferHandle ResourceAllocator::createBuffer(const vk::DeviceSize size_,
                                             const vk::BufferUsageFlags usage_,
                                             const MemoryMappingType mapping_type,
                                             const std::string& debug_name,
                                             const std::optional<vk::DeviceSize> min_alignment) {
    vk::BufferCreateInfo info{{}, size_, vk::BufferUsageFlagBits::eTransferDst | usage_};
    return createBuffer(info, mapping_type, debug_name, min_alignment);
}

BufferHandle ResourceAllocator::createBuffer(const CommandBufferHandle& cmdBuf,
                                             const vk::DeviceSize& size_,
                                             const vk::BufferUsageFlags usage_,
                                             const void* data_,
                                             const MemoryMappingType mapping_type,
                                             const std::string& debug_name,
                                             const std::optional<vk::DeviceSize> min_alignment) {
    BufferHandle resultBuffer =
        createBuffer(size_, usage_, mapping_type, debug_name, min_alignment);

    if (data_ != nullptr) {
        m_staging->cmd_to_device(cmdBuf, resultBuffer, data_);
    }

    return resultBuffer;
}

const BufferHandle& ResourceAllocator::get_dummy_buffer() const {
    return dummy_buffer;
}

// You get the alignment from
// VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment
BufferHandle ResourceAllocator::createScratchBuffer(const vk::DeviceSize size,
                                                    const vk::DeviceSize alignment,
                                                    const std::string& debug_name) {
    return createBuffer(size, Buffer::SCRATCH_BUFFER_USAGE, MemoryMappingType::NONE, debug_name,
                        alignment);
}

BufferHandle ResourceAllocator::createInstancesBuffer(const uint32_t instance_count,
                                                      const std::string& debug_name) {

    return createBuffer(sizeof(vk::AccelerationStructureInstanceKHR) * instance_count,
                        Buffer::INSTANCES_BUFFER_USAGE, merian::MemoryMappingType::NONE, debug_name,
                        16);
}

ImageHandle ResourceAllocator::createImage(const vk::ImageCreateInfo& info_,
                                           const MemoryMappingType mapping_type,
                                           const std::string& debug_name) {
    const ImageHandle image = m_memAlloc->create_image(info_, mapping_type, debug_name);

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->device, **image, debug_name);
    }
    SPDLOG_TRACE("created image {} ({})", fmt::ptr(static_cast<VkImage>(**image)), debug_name);
#endif

    return image;
}

ImageHandle ResourceAllocator::createImage(const CommandBufferHandle& cmdBuf,
                                           const void* data_,
                                           const vk::ImageCreateInfo& info_,
                                           const MemoryMappingType mapping_type,
                                           const std::string& debug_name) {
    assert(data_);

    const ImageHandle result_image = createImage(info_, mapping_type, debug_name);

    // doing these transitions per copy is not efficient, should do in bulk for many images
    cmdBuf->barrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                    result_image->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                                          vk::AccessFlagBits::eTransferWrite,
                                          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                          all_levels_and_layers(), true));

    m_staging->cmd_to_device(cmdBuf, result_image, data_);

    return result_image;
}

ImageHandle ResourceAllocator::createImageFromRGBA8(const CommandBufferHandle& cmd,
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
    return createImage(cmd, data, tex_image_info, MemoryMappingType::NONE, debug_name);
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
        debug_utils->set_object_name(context->device, **view, debug_name);
    }
    SPDLOG_TRACE("created image view {} ({}), for image {}",
                 fmt::ptr(static_cast<VkImageView>(**view)), debug_name,
                 fmt::ptr(static_cast<VkImage>(**image)));
#endif

    return view;
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& view_create_info,
                                               const SamplerHandle& sampler,
                                               [[maybe_unused]] const std::string& debug_name) {
    const ImageViewHandle view = create_image_view(image, view_create_info, debug_name);

    return Texture::create(view, sampler);
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& imageViewCreateInfo,
                                               const vk::SamplerCreateInfo& samplerCreateInfo,
                                               const std::string& debug_name) {
    const SamplerHandle sampler = m_samplerPool->acquire_sampler(samplerCreateInfo);
    return createTexture(image, imageViewCreateInfo, sampler, debug_name);
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const std::string& debug_name) {
    return createTexture(image, image->make_view_create_info(), debug_name);
}

TextureHandle ResourceAllocator::createTexture(const ImageHandle& image,
                                               const vk::ImageViewCreateInfo& view_create_info,
                                               const std::string& debug_name) {
    const ContextHandle& context = image->get_memory()->get_context();

    const vk::FormatProperties props =
        context->physical_device.physical_device.getFormatProperties(view_create_info.format);

    SamplerHandle sampler;
    if ((image->get_tiling() == vk::ImageTiling::eOptimal &&
         (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) ||
        (image->get_tiling() == vk::ImageTiling::eLinear &&
         (props.linearTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))) {
        sampler = get_sampler_pool()->linear_mirrored_repeat();
    } else {
        sampler = get_sampler_pool()->nearest_mirrored_repeat();
    }

    return createTexture(image, view_create_info, sampler, debug_name);
}

TextureHandle ResourceAllocator::createTextureFromRGBA8(const CommandBufferHandle& cmd,
                                                        const uint32_t* data,
                                                        const uint32_t width,
                                                        const uint32_t height,
                                                        const vk::Filter mag_filter,
                                                        const vk::Filter min_filter,
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

    const merian::ImageHandle image =
        createImageFromRGBA8(cmd, data, width, height, usage_flags, isSRGB, mip_levels, debug_name);

    if (generate_mipmaps) {
        for (uint32_t i = 1; i <= mip_levels; i++) {
            const vk::ImageMemoryBarrier bar{
                vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eTransferRead,
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eTransferSrcOptimal,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                *image,
                vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, i - 1, 1, 0, 1}};
            cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                         bar);
            // let the loop run one iteration more to get the whole image transitioned to transfer
            // src.
            if (i == mip_levels) {
                break;
            }

            vk::ImageBlit blit{
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i - 1, 0, 1},
                {},
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i, 0, 1},
                {}};
            blit.srcOffsets[1] =
                vk::Offset3D{int32_t(width >> (i - 1)), int32_t(height >> (i - 1)), 1};
            blit.dstOffsets[1] = vk::Offset3D{int32_t(width >> i), int32_t(height >> i), 1};
            cmd->blit(image, vk::ImageLayout::eTransferSrcOptimal, image,
                      vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);
        }
        image->_set_current_layout(vk::ImageLayout::eTransferSrcOptimal);
    }

    merian::SamplerHandle sampler = get_sampler_pool()->for_filter_and_address_mode(
        mag_filter, min_filter, vk::SamplerAddressMode::eRepeat);

    return createTexture(image, image->make_view_create_info(), sampler, debug_name);
}

const TextureHandle& ResourceAllocator::get_dummy_texture() const {
    return dummy_texture;
}

HWAccelerationStructureHandle ResourceAllocator::createAccelerationStructure(
    const vk::AccelerationStructureTypeKHR type,
    const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
    const std::string& debug_name) {
    // Allocating the buffer to hold the acceleration structure
    BufferHandle buffer = createBuffer(size_info.accelerationStructureSize,
                                       vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                           vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                       MemoryMappingType::NONE, debug_name);
    vk::AccelerationStructureKHR as;
    // Setting the buffer
    vk::AccelerationStructureCreateInfoKHR createInfo{
        {}, *buffer, {}, size_info.accelerationStructureSize, type};
    check_result(context->device.createAccelerationStructureKHR(&createInfo, nullptr, &as),
                 "could not create acceleration structure");

#ifndef NDEBUG
    if (debug_utils) {
        debug_utils->set_object_name(context->device, **buffer, debug_name);
        debug_utils->set_object_name(context->device, as, debug_name);
    }
#endif

    return HWAccelerationStructure::create(as, buffer, size_info);
}

std::shared_ptr<StagingMemoryManager> ResourceAllocator::getStaging() {
    return m_staging;
}

const std::shared_ptr<StagingMemoryManager>& ResourceAllocator::getStaging() const {
    return m_staging;
}

} // namespace merian
