#pragma once

#include "any_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

#include <any>
#include <memory>

namespace merian_nodes {

class AnyIn;
using AnyInHandle = std::shared_ptr<AnyIn>;

// Receive information from HostAnyOuts.
class AnyIn : public TypedInputConnector<AnyOutHandle, const std::any&> {

  public:
    AnyIn(const std::string& name, const uint32_t delay);

    const std::any& resource(const GraphResourceHandle& resource) override;

    void on_connect_output(const OutputConnectorHandle& output) override;

    Connector::ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const vk::CommandBuffer& cmd,
                    GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  public:
    static AnyInHandle create(const std::string& name, const uint32_t delay = 0);
};

} // namespace merian_nodes
