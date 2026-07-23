#pragma once

#include <memory>

#include "connector_access.hpp"
#include "graph_run.hpp"

#include "merian/shader/shader_cursor.hpp"
#include "merian/utils/properties.hpp"
#include "merian/vk/descriptors/descriptor_set.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

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
        // Signalize that a graph reconnect is required, for example to recreate all resoruces.
        NEEDS_RECONNECT = 0b1,
    };

  public:
    Connector() = default;

    virtual ~Connector() {};

    // Whether NodeIO::bind writes this connector into a shader cursor field.
    virtual bool shader_bindable() const {
        return false;
    }

    // Write the resource into the shader cursor field named after this connector.
    // resource is null if an optional input is not connected - write a dummy then.
    // access is the declared ConnectorAccess of the port.
    virtual void bind([[maybe_unused]] ShaderCursor& cursor,
                      [[maybe_unused]] const GraphResourceHandle& resource,
                      [[maybe_unused]] const ResourceAllocatorHandle& allocator,
                      [[maybe_unused]] const ConnectorAccess& access) {
        throw std::runtime_error{"connector does not support shader binding"};
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
                   [[maybe_unused]] const CommandBufferHandle& cmd,
                   [[maybe_unused]] const GraphResourceHandle& resource,
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
                    [[maybe_unused]] const CommandBufferHandle& cmd,
                    [[maybe_unused]] const GraphResourceHandle& resource,
                    [[maybe_unused]] const NodeHandle& node,
                    [[maybe_unused]] std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    [[maybe_unused]] std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) {
        return {};
    }

    // Mainly to describe yourself
    virtual void properties([[maybe_unused]] Properties& config) {}
};

using ConnectorHandle = std::shared_ptr<Connector>;

/**
 * @brief      Mixin for connector which allow some access to the underlying resource.
 *
 * @tparam     ResourceAccessType  defines how nodes can access the underlying resource of this
 * connector. If the type is void, access is not possible.
 */
template <typename ResourceAccessType = void> class AccessibleConnector {
  public:
    AccessibleConnector() {}

    using resource_access_type = ResourceAccessType;

    virtual ResourceAccessType resource(const GraphResourceHandle& resource) = 0;
};

template <typename ResourceType, typename ResourceAccessType = void>
using AccessibleConnectorHandle = std::shared_ptr<AccessibleConnector<ResourceAccessType>>;

} // namespace merian
