#include "merian-nodes/connectors/buffer/vk_buffer_out_unmanaged.hpp"
#include "merian-nodes/connectors/buffer/vk_buffer_in.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/utils/pointer.hpp"

namespace merian {

UnmanagedVkBufferOut::UnmanagedVkBufferOut(const std::string& name,
                                           const uint32_t array_size,
                                           const vk::BufferUsageFlags buffer_usage_flags)
    : VkBufferOut(name, false, array_size), buffer_usage_flags(buffer_usage_flags) {}

GraphResourceHandle UnmanagedVkBufferOut::create_resource(
    [[maybe_unused]] const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
    const ResourceAllocatorHandle& /*allocator*/,
    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
    [[maybe_unused]] const uint32_t resource_index,
    [[maybe_unused]] const uint32_t ring_size) {

    vk::BufferUsageFlags all_buffer_usage_flags = buffer_usage_flags;
    vk::PipelineStageFlags2 input_pipeline_stages;
    vk::AccessFlags2 input_access_flags;

    for (const auto& [input_node, input] : inputs) {
        const auto& con_in = debugable_ptr_cast<VkBufferIn>(input);
        all_buffer_usage_flags |= con_in->get_usage_flags();
        input_pipeline_stages |= con_in->get_pipeline_stages();
        input_access_flags |= con_in->get_access_flags();
    }

    if (buffers.empty()) {
        buffers.resize(get_array_size());
    } else {
        for (const auto& buffer : buffers) {
            if (buffer &&
                (buffer->get_usage_flags() & all_buffer_usage_flags) != all_buffer_usage_flags) {
                throw graph_errors::invalid_connection{fmt::format(
                    "buffers set for the unmanaged output connector {} are missing some "
                    "usage flags for the new inputs.",
                    name)};
            }
        }
    }

    return std::make_shared<UnmanagedBufferArrayResource>(get_array_size(), all_buffer_usage_flags,
                                                          input_pipeline_stages, input_access_flags,
                                                          buffers);
}

UnmanagedBufferArrayResource& UnmanagedVkBufferOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<UnmanagedBufferArrayResource>(resource);
}

Connector::ConnectorStatusFlags UnmanagedVkBufferOut::on_pre_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
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

UnmanagedVkBufferOutHandle
UnmanagedVkBufferOut::create(const std::string& name,
                             const uint32_t array_size,
                             const vk::BufferUsageFlags buffer_usage_flags) {
    return std::make_shared<UnmanagedVkBufferOut>(name, array_size, buffer_usage_flags);
}

} // namespace merian
