#pragma once

#include "merian-nodes/connectors/any_in.hpp"
#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/graph/errors.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian-nodes/resources/host_any_resource.hpp"

#include <memory>

namespace merian_nodes {

class AnyOut;
using AnyOutHandle = std::shared_ptr<AnyOut>;

// Transfer information between nodes on the host using shared_ptr.
class AnyOut : public TypedOutputConnector<std::any&> {

  public:
    AnyOut(const std::string& name, const bool persistent)
        : TypedOutputConnector(name, !persistent), persistent(persistent) {}

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                    [[maybe_unused]] const ResourceAllocatorHandle& aliasing_allocator) override {

        for (auto& [node, input] : inputs) {
            // check compatibility

            const auto& casted_in = std::dynamic_pointer_cast<AnyIn>(input);
            if (!casted_in) {
                throw graph_errors::connector_error{
                    fmt::format("AnyOut {} cannot output to {} of node {}.", Connector::name,
                                input->name, node->name)};
            }
        }

        return std::make_shared<AnyResource>(persistent ? -1 : (int32_t)inputs.size());
    }

    std::any& resource(const GraphResourceHandle& resource) override {
        return debugable_ptr_cast<AnyResource>(resource)->any;
    }

    Connector::ConnectorStatusFlags on_pre_process(
        [[maybe_unused]] GraphRun& run,
        [[maybe_unused]] const vk::CommandBuffer& cmd,
        GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override {
        const auto& res = debugable_ptr_cast<AnyResource>(resource);
        if (!persistent) {
            res->any.reset();
        }

        return {};
    }

    Connector::ConnectorStatusFlags on_post_process(
        [[maybe_unused]] GraphRun& run,
        [[maybe_unused]] const vk::CommandBuffer& cmd,
        GraphResourceHandle& resource,
        [[maybe_unused]] const NodeHandle& node,
        [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
        [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override {
        const auto& res = debugable_ptr_cast<AnyResource>(resource);
        if (!res->any.has_value()) {
            throw graph_errors::connector_error{fmt::format(
                "Node {} did not set the resource for output {}.", node->name, Connector::name)};
        }
        res->processed_inputs = 0;

        return {};
    }

  public:
    static AnyOutHandle create(const std::string& name, const bool persistent = false) {
        return std::make_shared<AnyOut>(name, persistent);
    }

  private:
    const bool persistent;
};

} // namespace merian_nodes
