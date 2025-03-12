#pragma once

#include "managed_vk_image_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

namespace merian_nodes {

class VkImageIn;
using VkImageInHandle = std::shared_ptr<VkImageIn>;

// Input a Vulkan image that is allocated and managed by the graph.
// Note that it only supplies a descriptor if stage_flags contains at least one bit.
class VkImageIn : public TypedInputConnector<ManagedVkImageOutHandle, ImageArrayResource&> {
    friend class ManagedVkImageOut;

  public:
    VkImageIn(const std::string& name,
                     const vk::AccessFlags2 access_flags,
                     const vk::PipelineStageFlags2 pipeline_stages,
                     const vk::ImageLayout required_layout,
                     const vk::ImageUsageFlags usage_flags,
                     const vk::ShaderStageFlags stage_flags,
                     const uint32_t delay = 0,
                     const bool optional = false);

    virtual std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    // For a optional input, resource can be nullptr here to signalize that no output was connected.
    // Provide a dummy binding in this case so that descriptor sets do not need to change.
    virtual void get_descriptor_update(const uint32_t binding,
                                       const GraphResourceHandle& resource,
                                       const DescriptorSetHandle& update,
                                       const ResourceAllocatorHandle& allocator) override;

    virtual ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const CommandBufferHandle& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    void on_connect_output(const OutputConnectorHandle& output) override;

    virtual ImageArrayResource& resource(const GraphResourceHandle& resource) override;

  public:
    static VkImageInHandle
    compute_read(const std::string& name, const uint32_t delay = 0, const bool optional = false);

    static VkImageInHandle
    transfer_src(const std::string& name, const uint32_t delay = 0, const bool optional = false);

    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
    const vk::ImageLayout required_layout;
    const vk::ImageUsageFlags usage_flags;
    const vk::ShaderStageFlags stage_flags;

    // set from output in create resource
    uint32_t array_size;
};

} // namespace merian_nodes
