#pragma once

#include "vk_buffer_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/resources/buffer_array_resource.hpp"

namespace merian {

class VkBufferIn;
using VkBufferInHandle = std::shared_ptr<VkBufferIn>;

class VkBufferIn : public InputConnector,
                   public OutputAccessibleInputConnector<VkBufferOutHandle>,
                   public AccessibleConnector<const BufferArrayResource&> {

  public:
    // A descriptor binding is only created if stage flags are supplied.
    VkBufferIn(const std::string& name,
               const vk::BufferUsageFlags usage_flags = {},
               const vk::ShaderStageFlags stage_flags = {},
               const vk::AccessFlags2 access_flags = {},
               const vk::PipelineStageFlags2 pipeline_stages = {},
               const uint32_t delay = 0,
               const bool optional = false);

    std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    void get_descriptor_update(const uint32_t binding,
                               const GraphResourceHandle& resource,
                               const DescriptorSetHandle& update,
                               const ResourceAllocatorHandle& allocator) override;

    void on_connect_output(const OutputConnectorHandle& output) override;

    ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const CommandBufferHandle& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    const BufferArrayResource& resource(const GraphResourceHandle& resource) override;

    // ---------------------------------

    uint32_t get_array_size() const {
        return array_size;
    }

    vk::BufferUsageFlags get_usage_flags() const {
        return usage_flags;
    }
    vk::ShaderStageFlags get_stage_flags() const {
        return stage_flags;
    }
    vk::AccessFlags2 get_access_flags() const {
        return access_flags;
    }
    vk::PipelineStageFlags2 get_pipeline_stages() const {
        return pipeline_stages;
    }

  public:
    static VkBufferInHandle
    compute_read(const std::string& name,
                 const uint32_t delay = 0,
                 const bool optional = false,
                 const vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer);

    static VkBufferInHandle
    fragment_read(const std::string& name,
                  const uint32_t delay = 0,
                  const bool optional = false,
                  const vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer);

    static VkBufferInHandle acceleration_structure_read(const std::string& name,
                                                        const uint32_t delay = 0,
                                                        const bool optional = false);

    static VkBufferInHandle
    transfer_src(const std::string& name, const uint32_t delay = 0, const bool optional = false);

  private:
    const vk::BufferUsageFlags usage_flags;
    const vk::ShaderStageFlags stage_flags;
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;

    // set from output in create resource
    uint32_t array_size = 1;
};

} // namespace merian
