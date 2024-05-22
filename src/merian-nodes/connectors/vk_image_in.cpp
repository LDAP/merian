#include "vk_image_in.hpp"
#include "merian-nodes/resources/vk_image_resource.hpp"

namespace merian_nodes {

VkImageIn::VkImageIn(const std::string& name,
                     const vk::AccessFlags2 access_flags,
                     const vk::PipelineStageFlags2 pipeline_stages,
                     const vk::ImageLayout required_layout,
                     const vk::ImageUsageFlags usage_flags,
                     const vk::ShaderStageFlags stage_flags,
                     const uint32_t delay)
    : TypedInputConnector(name, delay), access_flags(access_flags), pipeline_stages(pipeline_stages),
      required_layout(required_layout), usage_flags(usage_flags), stage_flags(stage_flags) {}

std::optional<vk::DescriptorSetLayoutBinding> VkImageIn::get_descriptor_info() const {
    return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler, 1, stage_flags, nullptr};
}

void VkImageIn::get_descriptor_update(const uint32_t binding,
                                      GraphResourceHandle& resource,
                                      DescriptorSetUpdate& update) {
    // // or vk::ImageLayout::eShaderReadOnlyOptimal instead of required?
    update.write_descriptor_texture(binding, this->resource(resource), 0, 1, required_layout);
}

Connector::ConnectorStatusFlags VkImageIn::on_pre_process(GraphRun& run,
                                                          const vk::CommandBuffer& cmd,
                                                          GraphResourceHandle& resource,
                                                          const NodeHandle& node,
                                                          std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                                                          std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    auto res = debugable_ptr_cast<VkTextureResource>(resource);
    if (res->last_used_as_output) {

    } else {
    }
    return {};
}

TextureHandle VkImageIn::resource(GraphResourceHandle& resource) {
    return debugable_ptr_cast<VkTextureResource>(resource)->tex;
}

VkImageIn VkImageIn::compute_read(const std::string& name, const uint32_t delay) {}

VkImageIn VkImageIn::transfer_src(const std::string& name, const uint32_t delay) {}

} // namespace merian_nodes
