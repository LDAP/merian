#include "merian-nodes/connectors/managed_vk_buffer_out.hpp"

#include <merian-nodes/connectors/managed_vk_image_out.hpp>

namespace merian_nodes {
ManagedVkImageOut::ManagedVkImageOut(const std::string& name,
                                     const vk::AccessFlags2& access_flags,
                                     const vk::PipelineStageFlags2& pipeline_stages,
                                     const vk::ImageLayout& required_layout,
                                     const vk::ShaderStageFlags& stage_flags,
                                     const vk::ImageCreateInfo& create_info,
                                     const bool persistent)
    : VkImageOut(name, access_flags, pipeline_stages, required_layout, stage_flags, create_info, persistent) {}

std::shared_ptr<ManagedVkImageOut> ManagedVkImageOut::compute_write(const std::string& name,
                                                                    const vk::Format format,
                                                                    const vk::Extent3D extent,
                                                                    const bool persistent) {
    const vk::ImageCreateInfo create_info{
            {},
            extent.depth == 1 ? vk::ImageType::e2D : vk::ImageType::e3D,
            format,
            extent,
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage,
            vk::SharingMode::eExclusive,
            {},
            {},
            vk::ImageLayout::eUndefined,
        };

    return std::make_shared<ManagedVkImageOut>(
        name, vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageLayout::eGeneral, vk::ShaderStageFlagBits::eCompute, create_info, persistent);
}

std::shared_ptr<ManagedVkImageOut> ManagedVkImageOut::compute_write(const std::string& name,
                                                                    const vk::Format format,
                                                                    const uint32_t width,
                                                                    const uint32_t height,
                                                                    const uint32_t depth,
                                                                    const bool persistent) {
    return compute_write(name, format, {width, height, depth}, persistent);
}

std::shared_ptr<ManagedVkImageOut>
ManagedVkImageOut::compute_fragment_write(const std::string& name,
                                          const vk::Format format,
                                          const vk::Extent3D extent,
                                          const bool persistent) {
    const vk::ImageCreateInfo create_info{
            {},
            extent.depth == 1 ? vk::ImageType::e2D : vk::ImageType::e3D,
            format,
            extent,
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment,
            vk::SharingMode::eExclusive,
            {},
            {},
            vk::ImageLayout::eUndefined,
        };

    return std::make_shared<ManagedVkImageOut>(
        name, vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageLayout::eGeneral,
        vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eFragment, create_info,
        persistent);
}

std::shared_ptr<ManagedVkImageOut>
ManagedVkImageOut::compute_fragment_write(const std::string& name,
                                          const vk::Format format,
                                          const uint32_t width,
                                          const uint32_t height,
                                          const uint32_t depth,
                                          const bool persistent) {
    return compute_fragment_write(name, format, {width, height, depth}, persistent);
}

std::shared_ptr<ManagedVkImageOut> ManagedVkImageOut::compute_read_write(const std::string& name,
                                                                         const vk::Format format,
                                                                         const vk::Extent3D extent,
                                                                         const bool persistent) {
    const vk::ImageCreateInfo create_info{
            {},
            extent.depth == 1 ? vk::ImageType::e2D : vk::ImageType::e3D,
            format,
            extent,
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            vk::SharingMode::eExclusive,
            {},
            {},
            vk::ImageLayout::eUndefined,
        };

    return std::make_shared<ManagedVkImageOut>(
        name, vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eComputeShader, vk::ImageLayout::eGeneral,
        vk::ShaderStageFlagBits::eCompute, create_info, persistent);
}

std::shared_ptr<ManagedVkImageOut>
ManagedVkImageOut::compute_read_write_transfer_dst(const std::string& name,
                                                   const vk::Format format,
                                                   const vk::Extent3D extent,
                                                   const vk::ImageLayout layout,
                                                   const bool persistent) {
    const vk::ImageCreateInfo create_info{
            {},
            extent.depth == 1 ? vk::ImageType::e2D : vk::ImageType::e3D,
            format,
            extent,
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                vk::ImageUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive,
            {},
            {},
            vk::ImageLayout::eUndefined,
        };

    return std::make_shared<ManagedVkImageOut>(
        name,
        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead |
            vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer, layout,
        vk::ShaderStageFlagBits::eCompute, create_info, persistent);
}

std::shared_ptr<ManagedVkImageOut>
ManagedVkImageOut::compute_read_write_transfer_dst(const std::string& name,
                                                   const vk::Format format,
                                                   const uint32_t width,
                                                   const uint32_t height,
                                                   const uint32_t depth,
                                                   const vk::ImageLayout layout,
                                                   const bool persistent) {
    return compute_read_write_transfer_dst(name, format, {width, height, depth}, layout,
                                           persistent);
}

std::shared_ptr<ManagedVkImageOut> ManagedVkImageOut::transfer_write(const std::string& name,
                                                                     const vk::Format format,
                                                                     const vk::Extent3D extent,
                                                                     const bool persistent) {
    const vk::ImageCreateInfo create_info{
            {},
            extent.depth == 1 ? vk::ImageType::e2D : vk::ImageType::e3D,
            format,
            extent,
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive,
            {},
            {},
            vk::ImageLayout::eUndefined,
        };

    return std::make_shared<ManagedVkImageOut>(
        name, vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllTransfer,
        vk::ImageLayout::eTransferDstOptimal, vk::ShaderStageFlags(), create_info, persistent);
}

std::shared_ptr<ManagedVkImageOut> ManagedVkImageOut::transfer_write(const std::string& name,
                                                                     const vk::Format format,
                                                                     const uint32_t width,
                                                                     const uint32_t height,
                                                                     const uint32_t depth,
                                                                     const bool persistent) {
    return transfer_write(name, format, {width, height, depth}, persistent);
}
} // namespace merian-nodes