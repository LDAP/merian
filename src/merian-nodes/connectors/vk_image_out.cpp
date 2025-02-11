#include "merian-nodes/connectors/vk_image_out.hpp"

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

VkImageOut::VkImageOut(const std::string& name,
                                     const vk::AccessFlags2& access_flags,
                                     const vk::PipelineStageFlags2& pipeline_stages,
                                     const vk::ImageLayout& required_layout,
                                     const vk::ShaderStageFlags& stage_flags,
                                     const vk::ImageCreateInfo& create_info,
                                     const bool persistent)
    : TypedOutputConnector(name, !persistent), access_flags(access_flags),
      pipeline_stages(pipeline_stages), required_layout(required_layout), stage_flags(stage_flags),
      create_info(create_info), persistent(persistent) {}

std::optional<vk::DescriptorSetLayoutBinding> VkImageOut::get_descriptor_info() const {
    if (stage_flags) {
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageImage, 1, stage_flags,
                                              nullptr};
    }
    return std::nullopt;
}

void VkImageOut::get_descriptor_update(
    const uint32_t binding,
    const GraphResourceHandle& resource,
    const DescriptorSetHandle& update,
    [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
    // or vk::ImageLayout::eGeneral instead of required?
    assert(debugable_ptr_cast<ManagedVkImageResource>(resource)->tex && "missing usage flags?");
    // From Spec 14.1.1: The image subresources for a storage image must be in the
    // VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or VK_IMAGE_LAYOUT_GENERAL layout in order to access its
    // data in a shader.
    update->queue_descriptor_write_texture(
        binding, *debugable_ptr_cast<ManagedVkImageResource>(resource)->tex, 0,
        vk::ImageLayout::eGeneral);
}

Connector::ConnectorStatusFlags VkImageOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    Connector::ConnectorStatusFlags flags{};
    const auto& res = debugable_ptr_cast<ManagedVkImageResource>(resource);
    if (res->needs_descriptor_update) {
        flags |= NEEDS_DESCRIPTOR_UPDATE;
        res->needs_descriptor_update = false;
    }

    vk::ImageMemoryBarrier2 img_bar =
        res->image->barrier2(required_layout, res->current_access_flags, access_flags,
                             res->current_stage_flags, pipeline_stages, VK_QUEUE_FAMILY_IGNORED,
                             VK_QUEUE_FAMILY_IGNORED, all_levels_and_layers(), !persistent);

    image_barriers.push_back(img_bar);
    res->current_stage_flags = pipeline_stages;
    res->current_access_flags = access_flags;

    return flags;
}

Connector::ConnectorStatusFlags VkImageOut::on_post_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    debugable_ptr_cast<ManagedVkImageResource>(resource)->last_used_as_output = true;
    return {};
}

GraphResourceHandle VkImageOut::create_resource(
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

ImageHandle VkImageOut::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<ManagedVkImageResource>(resource)->image;
}



} // namespace merian_nodes
