#include "merian-nodes/connectors/vk_tlas_out.hpp"
#include "merian-nodes/connectors/vk_tlas_in.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/utils/pointer.hpp"

namespace merian {

VkTLASOut::VkTLASOut() : OutputConnector(true) {}

GraphResourceHandle
VkTLASOut::create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                           [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                           [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
                           [[maybe_unused]] const uint32_t resource_index,
                           [[maybe_unused]] const uint32_t ring_size) {

    vk::PipelineStageFlags2 read_stages;

    for (const auto& [input_node, input] : inputs) {
        const auto& con_in = debugable_ptr_cast<VkTLASIn>(input);
        read_stages |= con_in->pipeline_stages;
    }

    return std::make_shared<TLASResource>(read_stages);
}

TLASResource& VkTLASOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<TLASResource>(resource);
}

Connector::ConnectorStatusFlags
VkTLASOut::on_pre_process([[maybe_unused]] GraphRun& run,
                          [[maybe_unused]] const CommandBufferHandle& cmd,
                          const GraphResourceHandle& resource,
                          [[maybe_unused]] const NodeHandle& node,
                          [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                          [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TLASResource>(resource);
    res->tlas.reset();

    return {};
}

Connector::ConnectorStatusFlags VkTLASOut::on_post_process(
    [[maybe_unused]] GraphRun& run,
    [[maybe_unused]] const CommandBufferHandle& cmd,
    const GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TLASResource>(resource);

    if (!res->tlas) {
        throw graph_errors::connector_error{"Node must set the TLAS for connector"};
    }

    Connector::ConnectorStatusFlags flags{};
    if (res->last_tlas != res->tlas) {
        res->last_tlas = res->tlas;
        flags |= NEEDS_DESCRIPTOR_UPDATE;
    }

    if (res->input_pipeline_stages) {
        buffer_barriers.push_back(res->tlas->tlas_read_barrier2(res->input_pipeline_stages));
    }

    return flags;
}

VkTLASOutHandle VkTLASOut::create() {
    return std::make_shared<VkTLASOut>();
}

} // namespace merian
