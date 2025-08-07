#include "merian-nodes/connectors/buffer/vk_buffer_in.hpp"
#include "merian-nodes/graph/errors.hpp"

namespace merian_nodes {

VkBufferIn::VkBufferIn(const std::string& name,
                       const vk::BufferUsageFlags usage_flags,
                       const vk::ShaderStageFlags stage_flags,
                       const vk::AccessFlags2 access_flags,
                       const vk::PipelineStageFlags2 pipeline_stages,
                       const uint32_t delay,
                       const bool optional)
    : InputConnector(name, delay, optional), usage_flags(usage_flags), stage_flags(stage_flags),
      access_flags(access_flags), pipeline_stages(pipeline_stages) {

    assert(access_flags && pipeline_stages);
    assert(!stage_flags || (access_flags & vk::AccessFlagBits2::eShaderRead));
}

std::optional<vk::DescriptorSetLayoutBinding> VkBufferIn::get_descriptor_info() const {
    if (stage_flags) {
        return vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer,
                                              get_array_size(), stage_flags, nullptr};
    }
    return std::nullopt;
}

void VkBufferIn::get_descriptor_update(const uint32_t binding,
                                       const GraphResourceHandle& resource,
                                       const DescriptorSetHandle& update,
                                       const ResourceAllocatorHandle& allocator) {
    if (!resource) {
        // the optional connector was not connected
        for (uint32_t i = 0; i < get_array_size(); i++) {
            update->queue_descriptor_write_buffer(binding, allocator->get_dummy_buffer(), 0,
                                                  VK_WHOLE_SIZE, i);
        }

        return;
    }

    const auto& res = debugable_ptr_cast<BufferArrayResource>(resource);
    for (auto& pending_update : res->pending_updates) {
        const BufferHandle buffer = res->get_buffer(pending_update);
        if (buffer) {
            update->queue_descriptor_write_buffer(binding, buffer, 0, VK_WHOLE_SIZE,
                                                  pending_update);
        } else {
            update->queue_descriptor_write_buffer(binding, allocator->get_dummy_buffer(), 0,
                                                  VK_WHOLE_SIZE, pending_update);
        }
    }
}

const BufferArrayResource& VkBufferIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<const BufferArrayResource>(resource);
}

void VkBufferIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<VkBufferOut>(output);
    if (!casted_output) {
        throw graph_errors::invalid_connection{
            fmt::format("VkBufferIn {} cannot recive from {}.", name, output->name)};
    }

    array_size = casted_output->get_array_size();
}

Connector::ConnectorStatusFlags VkBufferIn::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    auto res = debugable_ptr_cast<BufferArrayResource>(resource);

    Connector::ConnectorStatusFlags flags{};
    if (!res->current_updates.empty()) {
        res->pending_updates.clear();
        std::swap(res->pending_updates, res->current_updates);
        flags |= NEEDS_DESCRIPTOR_UPDATE;
    }

    return flags;
}

VkBufferInHandle VkBufferIn::compute_read(const std::string& name,
                                          const uint32_t delay,
                                          const bool optional,
                                          const vk::BufferUsageFlags usage) {
    return std::make_shared<VkBufferIn>(
        name, usage, vk::ShaderStageFlagBits::eCompute, vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eComputeShader, delay, optional);
}

VkBufferInHandle VkBufferIn::fragment_read(const std::string& name,
                                           const uint32_t delay,
                                           const bool optional,
                                           const vk::BufferUsageFlags usage) {
    return std::make_shared<VkBufferIn>(
        name, usage, vk::ShaderStageFlagBits::eFragment, vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eFragmentShader, delay, optional);
}

VkBufferInHandle VkBufferIn::acceleration_structure_read(const std::string& name,
                                                         const uint32_t delay,
                                                         const bool optional) {
    return std::make_shared<VkBufferIn>(
        name, vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        vk::ShaderStageFlags{}, vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, delay, optional);
}

std::shared_ptr<VkBufferIn>
VkBufferIn::transfer_src(const std::string& name, const uint32_t delay, const bool optional) {
    return std::make_shared<VkBufferIn>(name, vk::BufferUsageFlagBits::eTransferSrc,
                                        vk::ShaderStageFlags(), vk::AccessFlagBits2::eTransferRead,
                                        vk::PipelineStageFlagBits2::eAllTransfer, delay, optional);
}

} // namespace merian_nodes
