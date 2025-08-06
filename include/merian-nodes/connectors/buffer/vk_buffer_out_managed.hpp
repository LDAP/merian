#pragma once

#include "merian-nodes/connectors/buffer/vk_buffer_out.hpp"
#include "merian-nodes/resources/buffer_array_resource.hpp"

namespace merian_nodes {

class ManagedVkBufferOut;
using ManagedVkBufferOutHandle = std::shared_ptr<ManagedVkBufferOut>;

// Output a Vulkan image that is allocated and managed by the graph.
// Note that it only supplies a descriptor if stage_flags contains at least one bit.
class ManagedVkBufferOut : public VkBufferOut,
                           public AccessibleConnector<const BufferArrayResource&> {
  public:
    ManagedVkBufferOut(const std::string& name,
                       const vk::AccessFlags2& access_flags,
                       const vk::PipelineStageFlags2& pipeline_stages,
                       const vk::ShaderStageFlags& stage_flags,
                       const vk::BufferCreateInfo& create_info,
                       const bool persistent = false,
                       const uint32_t array_size = 1);

    virtual std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

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

    virtual ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const CommandBufferHandle& cmd,
                    const GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    virtual GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resource_index,
                    const uint32_t ring_size) override;

    virtual BufferArrayResource& resource(const GraphResourceHandle& resource) override;


  public:
    static ManagedVkBufferOutHandle compute_write(const std::string& name,
                                                  const vk::BufferCreateInfo& create_info,
                                                  const bool persistent = false);

    static ManagedVkBufferOutHandle transfer_write(const std::string& name,
                                                   const vk::BufferCreateInfo& create_info,
                                                   const bool persistent = false);

  private:

    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
    const vk::ShaderStageFlags stage_flags;
    const vk::BufferCreateInfo create_info;
    const bool persistent;
};

} // namespace merian_nodes
