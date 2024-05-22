#pragma once

#include "vk_image_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

namespace merian_nodes {

// Input a Vulkan image that is allocated and managed by the graph.
class VkImageIn : public TypedInputConnector<VkImageOut, TextureHandle> {
    friend class VkImageOut;

  public:
    VkImageIn(const std::string& name,
              const vk::AccessFlags2 access_flags,
              const vk::PipelineStageFlags2 pipeline_stages,
              const vk::ImageLayout required_layout,
              const vk::ImageUsageFlags usage_flags,
              const vk::ShaderStageFlags stage_flags,
              const uint32_t delay = 0);

    virtual std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    virtual void
    get_descriptor_update(const uint32_t binding, GraphResourceHandle& resource, DescriptorSetUpdate& update) override;

    virtual ConnectorStatusFlags on_pre_process(GraphRun& run,
                                                const vk::CommandBuffer& cmd,
                                                GraphResourceHandle& resource,
                                                const NodeHandle& node,
                                                std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                                                std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    virtual TextureHandle resource(GraphResourceHandle& resource) override;

  public:
    static VkImageIn compute_read(const std::string& name, const uint32_t delay = 0);

    static VkImageIn transfer_src(const std::string& name, const uint32_t delay = 0);

  private:
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
    const vk::ImageLayout required_layout;
    const vk::ImageUsageFlags usage_flags;
    const vk::ShaderStageFlags stage_flags;
};

} // namespace merian_nodes
