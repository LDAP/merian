#pragma once

#include "vk_buffer_out.hpp"

#include "merian-graph/graph/connector_input.hpp"
#include "merian-graph/resources/buffer_array_resource.hpp"

namespace merian {

class VkBufferIn;
using VkBufferInHandle = std::shared_ptr<VkBufferIn>;

// Receives a buffer (array). Declare how the node accesses it via ConnectorAccess in the
// descriptor; the graph synchronizes.
class VkBufferIn : public InputConnector,
                   public OutputAccessibleInputConnector<VkBufferOutHandle>,
                   public AccessibleConnector<const BufferArrayResource&> {

  public:
    VkBufferIn() = default;

    void on_connect_output(const OutputConnectorHandle& output) override;

    const BufferArrayResource& resource(const GraphResourceHandle& resource) override;

    bool shader_bindable() const override {
        return true;
    }

    void bind(ShaderCursor& cursor,
              const GraphResourceHandle& resource,
              const ResourceAllocatorHandle& allocator,
              const ConnectorAccess& access) override;

    uint32_t get_array_size() const {
        return array_size;
    }

  public:
    static VkBufferInHandle create();

  private:
    // set from output in on_connect_output
    uint32_t array_size = 1;
};

} // namespace merian
