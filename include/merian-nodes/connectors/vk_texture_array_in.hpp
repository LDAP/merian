#pragma once

#include "vk_texture_array_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

namespace merian_nodes {

class VkTextureArrayIn;
using VkTextureArrayInHandle = std::shared_ptr<VkTextureArrayIn>;

class VkTextureArrayIn
    : public TypedInputConnector<VkTextureArrayOutHandle, const TextureArrayResource&> {
    friend class VkTextureArrayOut;

  public:
    // A descriptor binding is only created if stage flags are supplied.
    VkTextureArrayIn(const std::string& name,
                     const vk::ShaderStageFlags stage_flags = {},
                     const vk::ImageLayout required_layout = {},
                     const vk::AccessFlags2 access_flags = {},
                     const vk::PipelineStageFlags2 pipeline_stages = {});

    std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    void get_descriptor_update(const uint32_t binding,
                               const GraphResourceHandle& resource,
                               const DescriptorSetHandle& update,
                               const ResourceAllocatorHandle& allocator) override;

    ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const CommandBufferHandle& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    const TextureArrayResource& resource(const GraphResourceHandle& resource) override;

    void on_connect_output(const OutputConnectorHandle& output) override;

  public:
    static VkTextureArrayInHandle compute_read(const std::string& name);

    static VkTextureArrayInHandle fragment_read(const std::string& name);

  private:
    const vk::ShaderStageFlags stage_flags;
    const vk::ImageLayout required_layout;
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;

    // set from output in create resource
    uint32_t array_size;
};

} // namespace merian_nodes
