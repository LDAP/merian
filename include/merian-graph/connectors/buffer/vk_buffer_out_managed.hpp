#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out.hpp"
#include "merian-graph/resources/buffer_array_resource.hpp"

namespace merian {

class ManagedVkBufferOut;
using ManagedVkBufferOutHandle = std::shared_ptr<ManagedVkBufferOut>;

// Output a buffer that is allocated and managed by the graph. Usage flags are the create info's
// usage plus the union of the declared ConnectorAccess of this port and all connected inputs.
class ManagedVkBufferOut : public VkBufferOut,
                           public AccessibleConnector<const BufferArrayResource&> {
  public:
    ManagedVkBufferOut(const vk::BufferCreateInfo& create_info,
                       const bool persistent = false,
                       const uint32_t array_size = 1);

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ConnectorAccess& combined_access,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resource_index,
                    const uint32_t ring_size) override;

    BufferArrayResource& resource(const GraphResourceHandle& resource) override;

    bool shader_bindable() const override {
        return true;
    }

    void bind(ShaderCursor& cursor,
              const GraphResourceHandle& resource,
              const ResourceAllocatorHandle& allocator,
              const ConnectorAccess& access) override;

  public:
    static ManagedVkBufferOutHandle create(const vk::BufferCreateInfo& create_info,
                                           const bool persistent = false,
                                           const uint32_t array_size = 1);

  public:
    const vk::BufferCreateInfo create_info;
    const bool persistent;
};

} // namespace merian
