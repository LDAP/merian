#pragma once

#include "merian-graph/graph/connector_output.hpp"
#include "merian-graph/graph/node.hpp"

#include <memory>

namespace merian {

class AnyOut;
using AnyOutHandle = std::shared_ptr<AnyOut>;

// Transfer information between nodes on the host using shared_ptr.
class AnyOut : public OutputConnector, public AccessibleConnector<std::any&> {
  public:
    bool supports_in_place() const override {
        return true;
    }

    GraphResourceHandle create_resource(const ResourceCreateInfo& info) override;

    std::any& resource(const GraphResourceHandle& resource) override;

    Connector::ConnectorStatusFlags on_pre_process(
        [[maybe_unused]] Submission& submission,
        const GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    Connector::ConnectorStatusFlags on_post_process(
        [[maybe_unused]] Submission& submission,
        const GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  public:
    static AnyOutHandle create();
};

} // namespace merian
