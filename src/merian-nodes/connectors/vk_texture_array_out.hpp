#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/resources/texture_array_resource.hpp"

namespace merian_nodes {

class TextureArrayOut;
using TextureArrayOutHandle = std::shared_ptr<TextureArrayOut>;

// Output an array of textures to use in a shader.
//
// When you set the texture the layout is automatically converted to eShaderReadOnlyOptimal after
// Node::process was called.
//
// The output keeps the textures alive for all in-flight iterations.
class TextureArrayOut : public TypedOutputConnector<TextureArrayResource&> {
    friend class TextureArrayIn;

  public:
    // A descriptor binding is only created if stage flags are supplied.
    // Note that you are responsible to convert the image to the layout that is required for the
    // descriptor binding.
    TextureArrayOut(const std::string& name,
                    const uint32_t array_size,
                    const std::optional<vk::ShaderStageFlags>& stage_flags = {});

    std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    void get_descriptor_update(const uint32_t binding,
                               GraphResourceHandle& resource,
                               DescriptorSetUpdate& update) override;

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resoruce_index,
                    const uint32_t ring_size) override;

    TextureArrayResource& resource(const GraphResourceHandle& resource) override;

    ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const vk::CommandBuffer& cmd,
                   GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const vk::CommandBuffer& cmd,
                    GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  public:
    static TextureArrayOutHandle create(const std::string& name, const uint32_t array_size);

  private:
    const uint32_t array_size;
    const std::optional<vk::ShaderStageFlags> stage_flags;
};

} // namespace merian_nodes
