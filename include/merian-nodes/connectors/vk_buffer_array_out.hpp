#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/resources/buffer_array_resource.hpp"

namespace merian_nodes {

class VkBufferArrayOut;
using VkBufferArrayOutHandle = std::shared_ptr<VkBufferArrayOut>;

// Output an array of buffers to use in a shader.
//
// Note that this connector does also persist the buffers accross graph rebuilds and it does
// set all descriptor slots to a dummy buffer (ResourceAllocator::get_dummy_buffer()) if not set.
//
// The output keeps the buffers alive for all in-flight iterations.
class VkBufferArrayOut : public TypedOutputConnector<BufferArrayResource&> {
    friend class VkBufferArrayIn;

  public:
    // No descriptor binding is created.
    VkBufferArrayOut(const std::string& name, const uint32_t array_size);

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resoruce_index,
                    const uint32_t ring_size) override;

    BufferArrayResource& resource(const GraphResourceHandle& resource) override;

    ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const CommandBufferHandle& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const CommandBufferHandle& cmd,
                    const GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    uint32_t array_size() const;

  public:
    static VkBufferArrayOutHandle create(const std::string& name, const uint32_t array_size);

  private:
    std::vector<merian::BufferHandle> buffers;
};

} // namespace merian_nodes
