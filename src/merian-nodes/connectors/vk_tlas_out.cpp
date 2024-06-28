#include "vk_tlas_out.hpp"
#include "vk_tlas_in.hpp"

#include "graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/utils/pointer.hpp"

namespace merian_nodes {

VkTLASOut::VkTLASOut(const std::string& name) : TypedOutputConnector(name, true) {}

GraphResourceHandle
VkTLASOut::create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                           [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                           [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
                           [[maybe_unused]] const uint32_t resoruce_index,
                           const uint32_t ring_size) {

    vk::PipelineStageFlags2 read_stages;

    for (auto& [input_node, input] : inputs) {
        const auto& con_in = std::dynamic_pointer_cast<VkTLASIn>(input);
        if (!con_in) {
            throw graph_errors::connector_error{
                fmt::format("VkTLASOut {} cannot output to {}.", name, input->name)};
        }
        read_stages |= con_in->pipeline_stages;
    }

    return std::make_shared<TLASResource>(read_stages, ring_size);
}

TLASResource& VkTLASOut::resource(const GraphResourceHandle& resource) {
    return *debugable_ptr_cast<TLASResource>(resource);
}

Connector::ConnectorStatusFlags
VkTLASOut::on_pre_process([[maybe_unused]] GraphRun& run,
                          [[maybe_unused]] const vk::CommandBuffer& cmd,
                          GraphResourceHandle& resource,
                          [[maybe_unused]] const NodeHandle& node,
                          [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                          [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TLASResource>(resource);
    res->tlas.reset();

    return {};
}

Connector::ConnectorStatusFlags VkTLASOut::on_post_process(
    GraphRun& run,
    [[maybe_unused]] const vk::CommandBuffer& cmd,
    GraphResourceHandle& resource,
    [[maybe_unused]] const NodeHandle& node,
    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {

    const auto& res = debugable_ptr_cast<TLASResource>(resource);

    if (!res->tlas) {
        throw graph_errors::connector_error{
            fmt::format("Node {} must set the TLAS for connector {}", node->name, name)};
    }

    Connector::ConnectorStatusFlags flags{};
    if (res->last_tlas != res->tlas) {
        res->last_tlas = res->tlas;
        flags |= NEEDS_DESCRIPTOR_UPDATE;
    }

    buffer_barriers.push_back(res->tlas->tlas_read_barrier2(res->input_pipeline_stages));
    res->in_flight_tlas[run.get_in_flight_index()] = res->tlas;

    return flags;
}

VkTLASOutHandle VkTLASOut::create(const std::string& name) {
    return std::make_shared<VkTLASOut>(name);
}

} // namespace merian_nodes
