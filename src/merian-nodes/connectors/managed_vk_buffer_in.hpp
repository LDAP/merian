#pragma once

#include "managed_vk_buffer_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

namespace merian_nodes {

class ManagedVkBufferIn;
using ManagedVkBufferInHandle = std::shared_ptr<ManagedVkBufferIn>;

// Input a Vulkan buffer that is allocated and managed by the graph.
// Note that it only supplies a descriptor if stage_flags contains at least one bit.
class ManagedVkBufferIn : public TypedInputConnector<ManagedVkBufferOutHandle, BufferHandle> {
    friend class ManagedVkBufferOut;

  public:
    ManagedVkBufferIn(const std::string& name,
                      const vk::AccessFlags2& access_flags,
                      const vk::PipelineStageFlags2& pipeline_stages,
                      const vk::BufferUsageFlags& usage_flags,
                      const vk::ShaderStageFlags& stage_flags,
                      const uint32_t delay = 0);

    virtual std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    virtual void get_descriptor_update(const uint32_t binding,
                                       const GraphResourceHandle& resource,
                                       DescriptorSetUpdate& update,
                                       const ResourceAllocatorHandle& allocator) override;

    virtual void on_connect_output(const OutputConnectorHandle& output) override;

    virtual BufferHandle resource(const GraphResourceHandle& resource) override;

    virtual ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const vk::CommandBuffer& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  public:
    static ManagedVkBufferInHandle compute_read(const std::string& name, const uint32_t delay = 0);

    static ManagedVkBufferInHandle transfer_src(const std::string& name, const uint32_t delay = 0);

  private:
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
    const vk::BufferUsageFlags usage_flags;
    const vk::ShaderStageFlags stage_flags;
};

} // namespace merian_nodes
