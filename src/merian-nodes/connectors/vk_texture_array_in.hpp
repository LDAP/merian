#pragma once

#include "vk_texture_array_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

namespace merian_nodes {

class TextureArrayIn;
using TextureArrayInHandle = std::shared_ptr<TextureArrayIn>;

class TextureArrayIn
    : public TypedInputConnector<TextureArrayOutHandle, const TextureArrayResource&> {
    friend class TextureArrayOut;

  public:
    // A descriptor binding is only created if stage flags are supplied.
    TextureArrayIn(const std::string& name,
                   const vk::ShaderStageFlags stage_flags = {},
                   const vk::ImageLayout required_layout = {},
                   const vk::AccessFlags2 access_flags = {},
                   const vk::PipelineStageFlags2 pipeline_stages = {});

    std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    void get_descriptor_update(const uint32_t binding,
                               GraphResourceHandle& resource,
                               DescriptorSetUpdate& update) override;

    ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const vk::CommandBuffer& cmd,
                   GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    const TextureArrayResource& resource(const GraphResourceHandle& resource) override;

    void on_connect_output(const OutputConnectorHandle& output) override;

    static TextureArrayInHandle compute_read(const std::string& name);

  private:
    const vk::ShaderStageFlags stage_flags;
    const vk::ImageLayout required_layout;
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;

    // set from output in create resource
    uint32_t array_size;
};

} // namespace merian_nodes
