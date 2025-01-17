#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/resources/texture_array_resource.hpp"

namespace merian_nodes {

class VkTextureArrayOut;
using VkTextureArrayOutHandle = std::shared_ptr<VkTextureArrayOut>;

// Output an array of textures.
//
// Note that this connector does also persist the textures accross graph rebuilds and it does
// set all descriptor slots to a dummy texture (ResourceAllocator::get_dummy_texture()) if not set.
//
// The output keeps the textures alive for all in-flight iterations.
class VkTextureArrayOut : public TypedOutputConnector<TextureArrayResource&> {
    friend class VkTextureArrayIn;

  public:
    // No descriptor binding is created.
    VkTextureArrayOut(const std::string& name, const uint32_t array_size);

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resoruce_index,
                    const uint32_t ring_size) override;

    TextureArrayResource& resource(const GraphResourceHandle& resource) override;

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
    static VkTextureArrayOutHandle create(const std::string& name, const uint32_t array_size);

  private:
    std::vector<merian::TextureHandle> textures;
};

} // namespace merian_nodes
