#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/resources/managed_vk_image_resource.hpp"

namespace merian_nodes {

class ManagedVkImageOut;
using ManagedVkImageOutHandle = std::shared_ptr<ManagedVkImageOut>;

// Output a Vulkan image that is allocated and managed by the graph.
// Note that it only supplies a descriptor if stage_flags contains at least one bit.
class ManagedVkImageOut : public TypedOutputConnector<ImageHandle> {
  public:
    ManagedVkImageOut(const std::string& name,
                      const vk::AccessFlags2& access_flags,
                      const vk::PipelineStageFlags2& pipeline_stages,
                      const vk::ImageLayout& required_layout,
                      const vk::ShaderStageFlags& stage_flags,
                      const vk::ImageCreateInfo& create_info,
                      const bool persistent = false);

    virtual std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    virtual void get_descriptor_update(const uint32_t binding,
                                       const GraphResourceHandle& resource,
                                       DescriptorSetUpdate& update,
                                       const ResourceAllocatorHandle& allocator) override;

    virtual ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const vk::CommandBuffer& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    virtual ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const vk::CommandBuffer& cmd,
                    const GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    virtual GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resoruce_index,
                    const uint32_t ring_size) override;

    virtual ImageHandle resource(const GraphResourceHandle& resource) override;

  public:
    static ManagedVkImageOutHandle compute_write(const std::string& name,
                                                 const vk::Format format,
                                                 const vk::Extent3D extent,
                                                 const bool persistent = false);

    static ManagedVkImageOutHandle compute_write(const std::string& name,
                                                 const vk::Format format,
                                                 const uint32_t width,
                                                 const uint32_t height,
                                                 const uint32_t depth = 1,
                                                 const bool persistent = false);

    static ManagedVkImageOutHandle
    compute_read_write_transfer_dst(const std::string& name,
                                    const vk::Format format,
                                    const vk::Extent3D extent,
                                    const vk::ImageLayout layout = vk::ImageLayout::eGeneral,
                                    const bool persistent = false);

    static ManagedVkImageOutHandle
    compute_read_write_transfer_dst(const std::string& name,
                                    const vk::Format format,
                                    const uint32_t width,
                                    const uint32_t height,
                                    const uint32_t depth = 1,
                                    const vk::ImageLayout layout = vk::ImageLayout::eGeneral,
                                    const bool persistent = false);

    static ManagedVkImageOutHandle compute_read_write(const std::string& name,
                                                      const vk::Format format,
                                                      const vk::Extent3D extent,
                                                      const bool persistent = false);

    static ManagedVkImageOutHandle transfer_write(const std::string& name,
                                                  const vk::Format format,
                                                  const vk::Extent3D extent,
                                                  const bool persistent = false);

    static ManagedVkImageOutHandle transfer_write(const std::string& name,
                                                  const vk::Format format,
                                                  const uint32_t width,
                                                  const uint32_t height,
                                                  const uint32_t depth = 1,
                                                  const bool persistent = false);

  public:
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
    const vk::ImageLayout required_layout;
    const vk::ShaderStageFlags stage_flags;
    const vk::ImageCreateInfo create_info;
    const bool persistent;
};

} // namespace merian_nodes
