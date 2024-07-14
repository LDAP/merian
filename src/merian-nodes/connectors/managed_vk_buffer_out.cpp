#include "managed_vk_buffer_out.hpp"

#include "merian-nodes/connectors/managed_vk_buffer_in.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/resources/managed_vk_buffer_resource.hpp"

#include "merian/utils/pointer.hpp"

namespace merian_nodes {

ManagedVkBufferOut::ManagedVkBufferOut(const std::string& name,
                                       const vk::AccessFlags2& access_flags,
                                       const vk::PipelineStageFlags2& pipeline_stages,
                                       const vk::ShaderStageFlags& stage_flags,
                                       const vk::BufferCreateInfo& create_info,
                                       const bool persistent)
    : TypedOutputConnector(name, !persistent), access_flags(access_flags),
      pipeline_stages(pipeline_stages), stage_flags(stage_flags), create_info(create_info),
      persistent(persistent) {}

std::optional<vk::DescriptorSetLayoutBinding> ManagedVkBufferOut::get_descriptor_info() const {
    if (stage_flags) {
        return vk::DescriptorSetLayoutBinding{
            0, vk::DescriptorType::eStorageBuffer, 1, stage_flags, nullptr,
        };
    } else {
        return std::nullopt;
    }
}

void ManagedVkBufferOut::get_descriptor_update(
    const uint32_t binding,
    const GraphResourceHandle& resource,
    DescriptorSetUpdate& update,
    [[maybe_unused]] const ResourceAllocatorHandle& allocator) {
    update.write_descriptor_buffer(binding,
                                   debugable_ptr_cast<ManagedVkBufferResource>(resource)->buffer);
}

Connector::ConnectorStatusFlags ManagedVkBufferOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    const auto& res = debugable_ptr_cast<ManagedVkBufferResource>(resource);

    vk::BufferMemoryBarrier2 buffer_bar{
        res->input_stage_flags,  res->input_access_flags, pipeline_stages, access_flags,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *res->buffer,    0,
        VK_WHOLE_SIZE,
    };
    buffer_barriers.push_back(buffer_bar);

    Connector::ConnectorStatusFlags flags{};
    if (res->needs_descriptor_update) {
        flags |= NEEDS_DESCRIPTOR_UPDATE;
        res->needs_descriptor_update = false;
    }

    return flags;
}

Connector::ConnectorStatusFlags ManagedVkBufferOut::on_post_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    const auto& res = debugable_ptr_cast<ManagedVkBufferResource>(resource);

    vk::BufferMemoryBarrier2 buffer_bar{
        pipeline_stages,
        access_flags,
        res->input_stage_flags,
        res->input_access_flags,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        *res->buffer,
        0,
        VK_WHOLE_SIZE,
    };
    buffer_barriers.push_back(buffer_bar);

    return {};
}

GraphResourceHandle ManagedVkBufferOut::create_resource(
    const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    const ResourceAllocatorHandle& allocator,
    const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resoruce_index,
    [[maybe_unused]] const uint32_t ring_size) {
    vk::BufferUsageFlags usage_flags = create_info.usage;
    vk::PipelineStageFlags2 input_pipeline_stages;
    vk::AccessFlags2 input_access_flags;

    if (!inputs.empty()) {
        for (auto& [input_node, input] : inputs) {
            const auto& buffer_in = debugable_ptr_cast<ManagedVkBufferIn>(input);
            usage_flags |= buffer_in->usage_flags;
            input_pipeline_stages |= buffer_in->pipeline_stages;
            input_access_flags |= buffer_in->access_flags;
        }
    } else {
        // prevent write after write hazard validation error.
        input_pipeline_stages |= pipeline_stages;
        input_access_flags |= access_flags;
    }

    ResourceAllocatorHandle alloc = persistent ? allocator : aliasing_allocator;
    const BufferHandle buffer = alloc->createBuffer(create_info, MemoryMappingType::NONE, name);

    return std::make_shared<ManagedVkBufferResource>(buffer, input_pipeline_stages,
                                                     input_access_flags);
}

BufferHandle ManagedVkBufferOut::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<ManagedVkBufferResource>(resource)->buffer;
}

std::shared_ptr<ManagedVkBufferOut> ManagedVkBufferOut::compute_write(
    const std::string& name, const vk::BufferCreateInfo& create_info, const bool persistent) {
    return std::make_shared<ManagedVkBufferOut>(
        name, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        vk::PipelineStageFlagBits2::eComputeShader, vk::ShaderStageFlagBits::eCompute, create_info,
        persistent);
}

std::shared_ptr<ManagedVkBufferOut> ManagedVkBufferOut::transfer_write(
    const std::string& name, const vk::BufferCreateInfo& create_info, const bool persistent) {
    return std::make_shared<ManagedVkBufferOut>(name, vk::AccessFlagBits2::eTransferWrite,
                                                vk::PipelineStageFlagBits2::eAllTransfer,
                                                vk::ShaderStageFlags(), create_info, persistent);
}

} // namespace merian_nodes
