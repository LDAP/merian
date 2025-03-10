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

Connector::ConnectorStatusFlags VkImageOut::on_post_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    debugable_ptr_cast<ImageArrayResource>(resource)->last_used_as_output = true;
    return {};
}

ImageArrayResource& VkImageOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<ImageArrayResource>(resource);
}


} // namespace merian_nodes
