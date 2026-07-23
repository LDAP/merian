#pragma once

#include "vk_tlas_out.hpp"

#include "merian-graph/graph/connector_input.hpp"
#include "merian-graph/resources/tlas_resource.hpp"

namespace merian {

class VkTLASIn;
using VkTLASInHandle = std::shared_ptr<VkTLASIn>;

// Input a TLAS. Declare how the node accesses it via ConnectorAccess in the descriptor.
class VkTLASIn : public InputConnector,
                 public OutputAccessibleInputConnector<VkTLASOutHandle>,
                 public AccessibleConnector<const AccelerationStructureHandle&> {
  public:
    VkTLASIn();

    void on_connect_output(const OutputConnectorHandle& output) override;

    const AccelerationStructureHandle& resource(const GraphResourceHandle& resource) override;

    bool shader_bindable() const override {
        return true;
    }

    void bind(ShaderCursor& cursor,
              const GraphResourceHandle& resource,
              const ResourceAllocatorHandle& allocator,
              const ConnectorAccess& access) override;

  public:
    static VkTLASInHandle create();
};

} // namespace merian
