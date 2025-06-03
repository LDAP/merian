#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/connectors/vk_image_out.hpp"
#include "merian-nodes/connectors/vk_texture_in.hpp"

namespace merian_nodes {

class UnmanagedVkImageOut;
using UnmanagedVkImageOutHandle = std::shared_ptr<UnmanagedVkImageOut>;

// Output an array of textures.
//
// Note that this connector does also persist the textures accross graph rebuilds and it does
// set all descriptor slots to a dummy texture (ResourceAllocator::get_dummy_texture()) if not set.
//
// The output keeps the textures alive for all in-flight iterations.
class UnmanagedVkImageOut : public VkImageOut {
    friend class VkTextureArrayIn;

  public:
    // No descriptor binding is created.
    UnmanagedVkImageOut(const std::string& name, const uint32_t array_size);

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resoruce_index,
                    const uint32_t ring_size) override;

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

  public:
    static UnmanagedVkImageOutHandle create(const std::string& name, const uint32_t array_size);
};

} // namespace merian_nodes
