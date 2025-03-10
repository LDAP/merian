#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/resources/managed_vk_image_resource.hpp"

#include <merian-nodes/resources/image_array_resource.hpp>

namespace merian_nodes {

class ManagedVkImageOut;
using ManagedVkImageOutHandle = std::shared_ptr<ManagedVkImageOut>;

// Output a Vulkan image that is allocated and managed by the graph.
// Note that it only supplies a descriptor if stage_flags contains at least one bit.
class VkImageOut : public TypedOutputConnector<ImageArrayResource&> {
  public:
    VkImageOut(const std::string& name,
                      const vk::AccessFlags2& access_flags,
                      const vk::PipelineStageFlags2& pipeline_stages,
                      const vk::ImageLayout& required_layout,
                      const vk::ShaderStageFlags& stage_flags,
                      const vk::ImageCreateInfo& create_info,
                      const bool persistent = false);

    virtual std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    virtual void get_descriptor_update(const uint32_t binding,
                                       const GraphResourceHandle& resource,
                                       const DescriptorSetHandle& update,
                                       const ResourceAllocatorHandle& allocator) override;

    virtual ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const CommandBufferHandle& cmd,
                    const GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    ImageArrayResource& resource(const GraphResourceHandle& resource) override;

    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
    const vk::ImageLayout required_layout;
    const vk::ShaderStageFlags stage_flags;
    const vk::ImageCreateInfo create_info;
    const bool persistent;

protected:
    std::vector<ImageHandle> images;
};

} // namespace merian_nodes
