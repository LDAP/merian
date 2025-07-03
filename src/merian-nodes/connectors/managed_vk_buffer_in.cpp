#include "merian-nodes/connectors/managed_vk_buffer_in.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/resources/managed_vk_buffer_resource.hpp"
#include "merian/utils/pointer.hpp"

namespace merian_nodes {

ManagedVkBufferIn::ManagedVkBufferIn(const std::string& name,
                                     const vk::AccessFlags2& access_flags,
                                     const vk::PipelineStageFlags2& pipeline_stages,
                                     const vk::BufferUsageFlags& usage_flags,
                                     const vk::ShaderStageFlags& stage_flags,
                                     const uint32_t delay)
    : TypedInputConnector(name, delay), access_flags(access_flags),
      pipeline_stages(pipeline_stages), usage_flags(usage_flags), stage_flags(stage_flags) {}

std::optional<vk::DescriptorSetLayoutBinding> ManagedVkBufferIn::get_descriptor_info() const {
    if (stage_flags) {
        return vk::DescriptorSetLayoutBinding{
            0, vk::DescriptorType::eStorageBuffer, array_size, stage_flags, nullptr,
        };
    }
    return std::nullopt;
}

void ManagedVkBufferIn::get_descriptor_update(
    const uint32_t binding,
    const GraphResourceHandle& resource,
    const DescriptorSetHandle& update,
    [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
    const auto& res = debugable_ptr_cast<BufferArrayResource>(resource);
    for (uint32_t i = 0; i < array_size; i++) {
        update->queue_descriptor_write_buffer(
            binding, res->buffers[i]);
    }
}

Connector::ConnectorStatusFlags ManagedVkBufferIn::on_pre_process(
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

void ManagedVkBufferIn::on_connect_output(const OutputConnectorHandle& output) {
    auto casted_output = std::dynamic_pointer_cast<ManagedVkBufferOut>(output);
    if (!casted_output) {
        throw graph_errors::invalid_connection{
            fmt::format("ManagedVkBufferIn {} cannot recive from {}.", name, output->name)};
    }
    array_size = casted_output->array_size();
}

BufferArrayResource& ManagedVkBufferIn::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<BufferArrayResource>(resource);
}

std::shared_ptr<ManagedVkBufferIn> ManagedVkBufferIn::compute_read(const std::string& name,
                                                                   const uint32_t delay) {
    return std::make_shared<ManagedVkBufferIn>(
        name, vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eComputeShader,
        vk::BufferUsageFlagBits::eStorageBuffer, vk::ShaderStageFlagBits::eCompute, delay);
}

std::shared_ptr<ManagedVkBufferIn> ManagedVkBufferIn::transfer_src(const std::string& name,
                                                                   const uint32_t delay) {
    return std::make_shared<ManagedVkBufferIn>(
        name, vk::AccessFlagBits2::eTransferRead, vk::PipelineStageFlagBits2::eAllTransfer,
        vk::BufferUsageFlagBits::eTransferSrc, vk::ShaderStageFlags(), delay);
}

} // namespace merian_nodes
