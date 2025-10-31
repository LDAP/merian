#pragma once

#include "any_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

#include <any>
#include <memory>

namespace merian {

class AnyIn;
using AnyInHandle = std::shared_ptr<AnyIn>;

// Receive information from HostAnyOuts.
class AnyIn : public InputConnector,
              public OutputAccessibleInputConnector<AnyOutHandle>,
              public AccessibleConnector<const std::any&> {

  public:
    AnyIn(const std::string& name, const uint32_t delay);

    const std::any& resource(const GraphResourceHandle& resource) override;

    void on_connect_output(const OutputConnectorHandle& output) override;

    Connector::ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const CommandBufferHandle& cmd,
                    const GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  public:
    static AnyInHandle create(const std::string& name, const uint32_t delay = 0);
};

} // namespace merian
