#include "vk_buffer_in.hpp"

#include "merian-nodes/resources/vk_buffer_resource.hpp"
#include "merian/utils/pointer.hpp"

namespace merian_nodes {

VkBufferIn::VkBufferIn(const std::string& name,
                       const vk::AccessFlags2& access_flags,
                       const vk::PipelineStageFlags2& pipeline_stages,
                       const vk::BufferUsageFlags& usage_flags,
                       const vk::ShaderStageFlags& stage_flags,
                       const uint32_t delay)
    : TypedInputConnector(name, delay), access_flags(access_flags),
      pipeline_stages(pipeline_stages), usage_flags(usage_flags), stage_flags(stage_flags) {}

std::optional<vk::DescriptorSetLayoutBinding> VkBufferIn::get_descriptor_info() const {
    if (stage_flags) {
        return vk::DescriptorSetLayoutBinding{
            0, vk::DescriptorType::eStorageBuffer, 1, stage_flags, nullptr,
        };
    } else {
        return std::nullopt;
    }
}

void VkBufferIn::get_descriptor_update(const uint32_t binding,
                                       GraphResourceHandle& resource,
                                       DescriptorSetUpdate& update) {
    update.write_descriptor_buffer(binding, debugable_ptr_cast<VkBufferResource>(resource)->buffer);
}

BufferHandle VkBufferIn::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<VkBufferResource>(resource)->buffer;
}

std::shared_ptr<VkBufferIn> VkBufferIn::compute_read(const std::string& name,
                                                     const uint32_t delay) {
    return std::make_shared<VkBufferIn>(
        name, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::BufferUsageFlagBits::eStorageBuffer, vk::ShaderStageFlagBits::eCompute, delay);
}

std::shared_ptr<VkBufferIn> VkBufferIn::transfer_src(const std::string& name,
                                                     const uint32_t delay) {
    return std::make_shared<VkBufferIn>(
        name, vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eAllTransfer,
        vk::BufferUsageFlagBits::eTransferSrc, vk::ShaderStageFlags(), delay);
}

} // namespace merian_nodes
