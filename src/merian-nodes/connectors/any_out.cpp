#include "merian-nodes/connectors/any_out.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/resources/host_any_resource.hpp"

namespace merian_nodes {

AnyOut::AnyOut(const std::string& name, const bool persistent)
    : OutputConnector(name, !persistent), persistent(persistent) {}

GraphResourceHandle
AnyOut::create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                        [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                        [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator,
                        [[maybe_unused]] const uint32_t resource_index,
                        [[maybe_unused]] const uint32_t ring_size) {

    return std::make_shared<AnyResource>(persistent ? -1 : (int32_t)inputs.size());
}

std::any& AnyOut::resource(const GraphResourceHandle& resource) {
    return debugable_ptr_cast<AnyResource>(resource)->any;
}

Connector::ConnectorStatusFlags
AnyOut::on_pre_process([[maybe_unused]] GraphRun& run,
                       [[maybe_unused]] const CommandBufferHandle& cmd,
                       const GraphResourceHandle& resource,
                       [[maybe_unused]] const NodeHandle& node,
                       [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                       [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    const auto& res = debugable_ptr_cast<AnyResource>(resource);
    if (!persistent) {
        res->any.reset();
    }

    return {};
}

Connector::ConnectorStatusFlags
AnyOut::on_post_process([[maybe_unused]] GraphRun& run,
                        [[maybe_unused]] const CommandBufferHandle& cmd,
                        const GraphResourceHandle& resource,
                        [[maybe_unused]] const NodeHandle& node,
                        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
    const auto& res = debugable_ptr_cast<AnyResource>(resource);
    if (!res->any.has_value()) {
        throw graph_errors::connector_error{
            fmt::format("Node did not set the resource for output {}.", Connector::name)};
    }
    res->processed_inputs = 0;

    return {};
}

AnyOutHandle AnyOut::create(const std::string& name, const bool persistent) {
    return std::make_shared<AnyOut>(name, persistent);
}

} // namespace merian_nodes
