#include "merian-nodes/connectors/managed_vk_buffer_out.hpp"

#include <merian-nodes/connectors/managed_vk_image_out.hpp>
#include "merian-nodes/connectors/vk_image_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include <merian/utils/pointer.hpp>

namespace merian_nodes {
ManagedVkImageOut::ManagedVkImageOut(const std::string& name,
                                     const vk::AccessFlags2& access_flags,
                                     const vk::PipelineStageFlags2& pipeline_stages,
                                     const vk::ImageLayout& required_layout,
                                     const vk::ShaderStageFlags& stage_flags,
                                     const vk::ImageCreateInfo& create_info,
                                     const bool persistent,
                                     const uint32_t image_count)
        : VkImageOut(name, persistent, image_count), access_flags(access_flags), pipeline_stages(pipeline_stages),
        required_layout(required_layout), stage_flags(stage_flags), create_info(create_info), persistent(persistent) {}

std::optional<vk::DescriptorSetLayoutBinding> ManagedVkImageOut::get_descriptor_info() const {
    if (stage_flags) {
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageImage, 1, stage_flags,
                                              nullptr};
    }
    return std::nullopt;
}

void ManagedVkImageOut::get_descriptor_update(
    const uint32_t binding,
    const GraphResourceHandle& resource,
    const DescriptorSetHandle& update,
    [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
    // or vk::ImageLayout::eGeneral instead of required?
    if (!resource) {
        // the optional connector was not connected
        update->queue_descriptor_write_texture(binding, allocator->get_dummy_texture(), 0,
                                               vk::ImageLayout::eShaderReadOnlyOptimal);
    } else {
        const auto& res = debugable_ptr_cast<ImageArrayResource>(resource);
        for (auto& pending_update : res->pending_updates) {
            const TextureHandle tex =
                res->textures[pending_update].has_value() ? res->textures[pending_update].value() : allocator->get_dummy_texture();
            update->queue_descriptor_write_texture(binding, tex, pending_update, vk::ImageLayout::eGeneral);
        }
    }
    // From Spec 14.1.1: The image subresources for a storage image must be in the
    // VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or VK_IMAGE_LAYOUT_GENERAL layout in order to access its
    // data in a shader.

}

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
    vk::ImageLayout first_input_layout = vk::ImageLayout::eUndefined;

    std::map<std::pair<NodeHandle, uint32_t>, vk::ImageLayout> layouts_per_node;

    for (const auto& [input_node, input] : inputs) {
        const auto& image_in = debugable_ptr_cast<VkImageIn>(input);
        image_create_info.usage |= image_in->usage_flags;
        input_pipeline_stages |= image_in->pipeline_stages;
        input_access_flags |= image_in->access_flags;

        if (first_input_layout == vk::ImageLayout::eUndefined) {
            first_input_layout = image_in->required_layout;
        }

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

    for (auto & image : images) {
        image = alloc->createImage(image_create_info, MemoryMappingType::NONE, name);
    }

    auto res = std::make_shared<ImageArrayResource>(images, allocator->get_dummy_texture()->get_image(),
                                              input_pipeline_stages, input_access_flags,
                                              first_input_layout);

    for (uint32_t i = 0; i < res->images.size(); i++) {
        if (images[i]->valid_for_view()) {
            res->textures[i] = allocator->createTexture(images[i], images[i]->make_view_create_info(), name);
        }
    }

    return res;
}

Connector::ConnectorStatusFlags ManagedVkImageOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    auto res = debugable_ptr_cast<ImageArrayResource>(resource);

    for (auto& image : res->images) {
        vk::ImageMemoryBarrier2 img_bar =
        image->barrier2(required_layout, res->current_access_flags, access_flags,
                             res->current_stage_flags, pipeline_stages, VK_QUEUE_FAMILY_IGNORED,
                             VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), !persistent);
        image_barriers.push_back(img_bar);
    }

    res->current_stage_flags = pipeline_stages;
    res->current_access_flags = access_flags;

    Connector::ConnectorStatusFlags flags{};
    if (!res->current_updates.empty()) {
        res->pending_updates.clear();
        std::swap(res->pending_updates, res->current_updates);
        return NEEDS_DESCRIPTOR_UPDATE;
    }

    return flags;
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