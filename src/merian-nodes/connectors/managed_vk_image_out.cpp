#include "merian-nodes/connectors/managed_vk_buffer_out.hpp"

#include <merian-nodes/connectors/managed_vk_image_out.hpp>
#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include <merian/utils/pointer.hpp>

namespace merian_nodes {
ManagedVkImageOut::ManagedVkImageOut(const std::string& name,
                                     const vk::AccessFlags2& access_flags,
                                     const vk::PipelineStageFlags2& pipeline_stages,
                                     const vk::ImageLayout& required_layout,
                                     const vk::ShaderStageFlags& stage_flags,
                                     const vk::ImageCreateInfo& create_info,
                                     const bool persistent)
    : VkImageOut(name, access_flags, pipeline_stages, required_layout, stage_flags, create_info, persistent) {}

GraphResourceHandle ManagedVkImageOut::create_resource(
    const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    [[maybe_unused]] const ResourceAllocatorHandle& allocator,
    const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resource_index,
    [[maybe_unused]] const uint32_t ring_size) {
    const ResourceAllocatorHandle alloc = persistent ? allocator : aliasing_allocator;

    vk::ImageCreateInfo image_create_info = create_info;
    vk::PipelineStageFlags2 input_pipeline_stages;
    vk::AccessFlags2 input_access_flags;

    std::map<std::pair<NodeHandle, uint32_t>, vk::ImageLayout> layouts_per_node;

    for (const auto& [input_node, input] : inputs) {
        const auto& image_in = debugable_ptr_cast<ManagedVkImageIn>(input);
        image_create_info.usage |= image_in->usage_flags;
        input_pipeline_stages |= image_in->pipeline_stages;
        input_access_flags |= image_in->access_flags;

        if (layouts_per_node.contains(std::make_pair(input_node, input->delay)) &&
            layouts_per_node.at(std::make_pair(input_node, input->delay)) !=
                image_in->required_layout) {
            throw graph_errors::connector_error{
                fmt::format("node has two input descriptors (one is {}) pointing to the "
                            "same underlying resource with different image layouts.",
                            name)};
                }
        layouts_per_node.try_emplace(std::make_pair(input_node, input->delay),
                                     image_in->required_layout);
    }

    const ImageHandle image = alloc->createImage(image_create_info, MemoryMappingType::NONE, name);
    auto res =
        std::make_shared<ManagedVkImageResource>(image, input_pipeline_stages, input_access_flags);

    if (image->valid_for_view()) {
        res->tex = allocator->createTexture(image, image->make_view_create_info(), name);
    }

    return res;
}

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