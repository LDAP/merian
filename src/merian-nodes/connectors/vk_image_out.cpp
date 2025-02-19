#include "merian-nodes/connectors/vk_image_out.hpp"


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

ImageHandle VkImageOut::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<ManagedVkImageResource>(resource)->image;
}


} // namespace merian_nodes
