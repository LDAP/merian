#include "vk_tlas_in.hpp"

namespace merian_nodes {

VkTLASIn::VkTLASIn(const std::string& name, const vk::ShaderStageFlags stage_flags)
    : TypedInputConnector(name, 0) {
    assert(stage_flags);
}

std::optional<vk::DescriptorSetLayoutBinding> VkTLASIn::get_descriptor_info() const {
    return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eAccelerationStructureKHR, 1,
                                          stage_flags, nullptr};
}

void VkTLASIn::get_descriptor_update(const uint32_t binding,
                                     GraphResourceHandle& resource,
                                     DescriptorSetUpdate& update) {
    const auto& res = debugable_ptr_cast<TLASResource>(resource);
    update.write_descriptor_acceleration_structure(binding, *res->tlas);
}

const AccelerationStructureHandle& VkTLASIn::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<TLASResource>(resource)->tlas;
}

VkTLASInHandle VkTLASIn::compute_read(const std::string& name) {
    return std::make_shared<VkTLASIn>(name, vk::ShaderStageFlagBits::eCompute);
}

} // namespace merian_nodes
