#include "merian-nodes/connectors/vk_tlas_in.hpp"
#include "merian-nodes/graph/errors.hpp"

namespace merian {

VkTLASIn::VkTLASIn(const vk::ShaderStageFlags stage_flags,
                   const vk::PipelineStageFlags2 pipeline_stages)
    : InputConnector(0), stage_flags(stage_flags), pipeline_stages(pipeline_stages) {
    assert(stage_flags);
    assert(pipeline_stages);
}

void VkTLASIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<VkTLASOut>(output);
    if (!casted_output) {
        throw graph_errors::invalid_connection{"VkTLASIn cannot receive from output."};
    }
}

std::optional<vk::DescriptorSetLayoutBinding> VkTLASIn::get_descriptor_info() const {
    return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eAccelerationStructureKHR, 1,
                                          stage_flags, nullptr};
}

void VkTLASIn::get_descriptor_update(const uint32_t binding,
                                     const GraphResourceHandle& resource,
                                     const DescriptorSetHandle& update,
                                     [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
    const auto& res = debugable_ptr_cast<TLASResource>(resource);
    update->queue_descriptor_write_acceleration_structure(binding, res->tlas);
}

const AccelerationStructureHandle& VkTLASIn::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<TLASResource>(resource)->tlas;
}

VkTLASInHandle VkTLASIn::compute_read() {
    return std::make_shared<VkTLASIn>(vk::ShaderStageFlagBits::eCompute,
                                      vk::PipelineStageFlagBits2::eComputeShader);
}

VkTLASInHandle VkTLASIn::fragment_read() {
    return std::make_shared<VkTLASIn>(vk::ShaderStageFlagBits::eFragment,
                                      vk::PipelineStageFlagBits2::eFragmentShader);
}

} // namespace merian
