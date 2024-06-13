#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/graph/node.hpp"

#include <memory>

namespace merian_nodes {

class AnyOut;
using AnyOutHandle = std::shared_ptr<AnyOut>;

// Transfer information between nodes on the host using shared_ptr.
class AnyOut : public TypedOutputConnector<std::any&> {

  public:
    AnyOut(const std::string& name, const bool persistent);

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator) override;

    std::any& resource(const GraphResourceHandle& resource) override;

    Connector::ConnectorStatusFlags on_pre_process(
        [[maybe_unused]] GraphRun& run,
        [[maybe_unused]] const vk::CommandBuffer& cmd,
        GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    Connector::ConnectorStatusFlags on_post_process(
        [[maybe_unused]] GraphRun& run,
        [[maybe_unused]] const vk::CommandBuffer& cmd,
        GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  public:
    static AnyOutHandle create(const std::string& name, const bool persistent = false);

  private:
    const bool persistent;
};

} // namespace merian_nodes
