#pragma once

#include "vk_buffer_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"

namespace merian_nodes {

class VkBufferIn;
using VkBufferInHandle = std::shared_ptr<VkBufferIn>;

// Input a Vulkan buffer that is allocated and managed by the graph.
// Note that it only supplies a descriptor if stage_flags contains at least one bit.
class VkBufferIn : public TypedInputConnector<VkBufferOut, BufferHandle> {
    friend class VkBufferOut;

  public:
    VkBufferIn(const std::string& name,
               const vk::AccessFlags2& access_flags,
               const vk::PipelineStageFlags2& pipeline_stages,
               const vk::BufferUsageFlags& usage_flags,
               const vk::ShaderStageFlags& stage_flags,
               const uint32_t delay = 0);

    virtual std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    virtual void get_descriptor_update(const uint32_t binding,
                                       GraphResourceHandle& resource,
                                       DescriptorSetUpdate& update) override;

    virtual BufferHandle resource(const GraphResourceHandle& resource) override;

  public:
    static VkBufferInHandle compute_read(const std::string& name, const uint32_t delay = 0);

    static VkBufferInHandle transfer_src(const std::string& name, const uint32_t delay = 0);

  private:
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
    const vk::BufferUsageFlags usage_flags;
    const vk::ShaderStageFlags stage_flags;
};

} // namespace merian_nodes
