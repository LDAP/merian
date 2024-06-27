#pragma once

#include <memory>

#include "merian/utils/properties.hpp"
#include "merian/vk/descriptors/descriptor_set_update.hpp"

#include "graph_run.hpp"

namespace merian_nodes {

class Node;
using NodeHandle = std::shared_ptr<Node>;
class GraphResource;
using GraphResourceHandle = std::shared_ptr<GraphResource>;

// An IO connector for a Node. Connectors might be reused and should therefore only contain the
// minimal necessary state and put everything else into the resoruces.
class Connector : public std::enable_shared_from_this<Connector> {
  public:
    using ConnectorStatusFlags = uint32_t;

    enum ConnectorStatusFlagBits {
        // Signalize that the resource has changed and descriptor set updates are necessary.
        //  You can assume that after you return this falg the descriptor sets are updated (and
        //  you can reset needs_descriptor_update).
        //
        //  Not only the descriptor set for this connector but every descriptor set that accesses
        //  the resource is
        //  updated.
        NEEDS_DESCRIPTOR_UPDATE = 0b1,

        // Signalize that a graph reconnect is required, for example to recreate all resoruces.
        NEEDS_RECONNECT = 0b10,
    };

  public:
    Connector(const std::string& name) : name(name) {}

    virtual ~Connector(){};

    // If the resource should be available in a shader, return a DescriptorSetLayoutBinding.
    // Note, that the binding value is ignored!
    virtual std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const {
        return std::nullopt;
    }

    // Write the descriptor update to the specified binding (please).
    // This is only called if get_descriptor_info() != std::nullopt.
    //
    // Assume that the last updates are persisted and only changes need to be recorded.
    virtual void get_descriptor_update([[maybe_unused]] const uint32_t binding,
                                       [[maybe_unused]] GraphResourceHandle& resource,
                                       [[maybe_unused]] DescriptorSetUpdate& update) {
        throw std::runtime_error{"resource is not accessible using a descriptor"};
    }

    // Called right after the node with this connector has finished node.pre_process() and before
    // node.process(). This is the place to insert barriers, if necessary. Prefer adding your
    // barriers to the supplied vectors instead of adding them directlty to the command buffer (for
    // performance reasons).
    //
    // Also, you can validate here that the node did use the output correctly (set the resource in
    // pre_process or do not access the same image with different layouts for example) and throw
    // merian::graph_errors::connector_error if not.
    //
    // The graph supplies here the resource for the current iteration (depending on delay and such).
    [[nodiscard]]
    virtual ConnectorStatusFlags
    on_pre_process([[maybe_unused]] GraphRun& run,
                   [[maybe_unused]] const vk::CommandBuffer& cmd,
                   [[maybe_unused]] GraphResourceHandle& resource,
                   [[maybe_unused]] const NodeHandle& node,
                   [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
        return {};
    }

    // Called right after the node with this connector has finished node.process(). For example, you
    // can validate here that the node did use the output correctly (set the resource for example)
    // and throw merian::graph_errors::connector_error if not.
    //
    // The graph supplies here the resource for the current iteration (depending on delay and such).
    [[nodiscard]]
    virtual ConnectorStatusFlags
    on_post_process([[maybe_unused]] GraphRun& run,
                    [[maybe_unused]] const vk::CommandBuffer& cmd,
                    [[maybe_unused]] GraphResourceHandle& resource,
                    [[maybe_unused]] const NodeHandle& node,
                    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
        return {};
    }

    // Mainly to describe yourself
    virtual void properties([[maybe_unused]] Properties& config) {}

  public:
    const std::string name;
};

using ConnectorHandle = std::shared_ptr<Connector>;

} // namespace merian_nodes
