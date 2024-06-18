#pragma once

#include "vk_buffer_array_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

namespace merian_nodes {

class BufferArrayIn;
using BufferArrayInHandle = std::shared_ptr<BufferArrayIn>;

class BufferArrayIn : public TypedInputConnector<BufferArrayOutHandle, const BufferArrayResource&> {
    friend class BufferArrayOut;

  public:
    // A descriptor binding is only created if stage flags are supplied.
    BufferArrayIn(const std::string& name,
                  const vk::ShaderStageFlags stage_flags = {},
                  const vk::AccessFlags2 access_flags = {},
                  const vk::PipelineStageFlags2 pipeline_stages = {});

    std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    void get_descriptor_update(const uint32_t binding,
                               GraphResourceHandle& resource,
                               DescriptorSetUpdate& update) override;

    const BufferArrayResource& resource(const GraphResourceHandle& resource) override;

    void on_connect_output(const OutputConnectorHandle& output) override;

    static BufferArrayInHandle compute_read(const std::string& name);

  private:
    const vk::ShaderStageFlags stage_flags;
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;

    // set from output in create resource
    uint32_t array_size;
};

} // namespace merian_nodes
