#include "vk_tlas_in.hpp"

namespace merian_nodes {

VkTLASIn::VkTLASIn(const std::string& name,
                   const vk::ShaderStageFlags stage_flags,
                   const vk::PipelineStageFlags2 pipeline_stages)
    : TypedInputConnector(name, 0), stage_flags(stage_flags), pipeline_stages(pipeline_stages) {
    assert(stage_flags);
    assert(pipeline_stages);
}

std::optional<vk::DescriptorSetLayoutBinding> VkTLASIn::get_descriptor_info() const {
    return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eAccelerationStructureKHR, 1,
                                          stage_flags, nullptr};
}

void VkTLASIn::get_descriptor_update(const uint32_t binding,
                                     const GraphResourceHandle& resource,
                                     DescriptorSetUpdate& update,
                                     [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
    const auto& res = debugable_ptr_cast<TLASResource>(resource);
    update.write_descriptor_acceleration_structure(binding, *res->tlas);
}

const AccelerationStructureHandle& VkTLASIn::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<TLASResource>(resource)->tlas;
}

VkTLASInHandle VkTLASIn::compute_read(const std::string& name) {
    return std::make_shared<VkTLASIn>(name, vk::ShaderStageFlagBits::eCompute,
                                      vk::PipelineStageFlagBits2::eComputeShader);
}

} // namespace merian_nodes
