#include "merian-nodes/connectors/vk_buffer_array_out.hpp"
#include "merian-nodes/connectors/vk_buffer_array_in.hpp"

#include "merian-nodes/graph/node.hpp"
#include "merian/utils/pointer.hpp"

namespace merian_nodes {

VkBufferArrayOut::VkBufferArrayOut(const std::string& name, const uint32_t array_size)
    : TypedOutputConnector(name, false), buffers(array_size) {}

GraphResourceHandle VkBufferArrayOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    const ResourceAllocatorHandle& allocator,
    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resoruce_index,
    const uint32_t ring_size) {

    vk::PipelineStageFlags2 input_pipeline_stages;
    vk::AccessFlags2 input_access_flags;

    for (auto& [input_node, input] : inputs) {
        const auto& con_in = debugable_ptr_cast<VkBufferArrayIn>(input);
        input_pipeline_stages |= con_in->pipeline_stages;
        input_access_flags |= con_in->access_flags;
    }

    return std::make_shared<BufferArrayResource>(buffers, ring_size, allocator->get_dummy_buffer(),
                                                 input_pipeline_stages, input_access_flags);
}

BufferArrayResource& VkBufferArrayOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<BufferArrayResource>(resource);
}

Connector::ConnectorStatusFlags VkBufferArrayOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<BufferArrayResource>(resource);
    if (!res->current_updates.empty()) {
        res->pending_updates.clear();
        std::swap(res->pending_updates, res->current_updates);
        return NEEDS_DESCRIPTOR_UPDATE;
    }

    return {};
}

Connector::ConnectorStatusFlags VkBufferArrayOut::on_post_process(
    GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<BufferArrayResource>(resource);

    Connector::ConnectorStatusFlags flags{};
    if (!res->current_updates.empty()) {
        res->pending_updates.clear();
        std::swap(res->pending_updates, res->current_updates);
        flags |= NEEDS_DESCRIPTOR_UPDATE;
    }

    res->in_flight_buffers[run.get_in_flight_index()] = buffers;

    return flags;
}

VkBufferArrayOutHandle VkBufferArrayOut::create(const std::string& name,
                                                const uint32_t array_size) {
    return std::make_shared<VkBufferArrayOut>(name, array_size);
}

uint32_t VkBufferArrayOut::array_size() const {
    return buffers.size();
}

} // namespace merian_nodes
