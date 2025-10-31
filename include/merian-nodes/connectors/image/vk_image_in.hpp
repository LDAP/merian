#pragma once

#include "vk_image_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/resources/image_array_resource.hpp"

namespace merian {

class VkImageIn;
using VkImageInHandle = std::shared_ptr<VkImageIn>;

// Note that it only supplies a descriptor if stage_flags contains at least one bit.
class VkImageIn : public InputConnector,
                  public OutputAccessibleInputConnector<VkImageOutHandle>,
                  public AccessibleConnector<const ImageArrayResource&> {
  public:
    VkImageIn(const std::string& name,
              const vk::AccessFlags2 access_flags,
              const vk::PipelineStageFlags2 pipeline_stages,
              const vk::ImageLayout required_layout,
              const vk::ImageUsageFlags usage_flags,
              const vk::ShaderStageFlags stage_flags,
              const uint32_t delay = 0,
              const bool optional = false);

    virtual ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const CommandBufferHandle& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    void on_connect_output(const OutputConnectorHandle& output) override;

    virtual const ImageArrayResource& resource(const GraphResourceHandle& resource) override;

    // ------------------------------------------
    vk::AccessFlags2 get_access_flags() const {
        return access_flags;
    }

    vk::PipelineStageFlags2 get_pipeline_stages() const {
        return pipeline_stages;
    }

    vk::ImageLayout get_required_layout() const {
        return required_layout;
    }

    vk::ImageUsageFlags get_usage_flags() const {
        return usage_flags;
    }

    vk::ShaderStageFlags get_stage_flags() const {
        return stage_flags;
    }

    uint32_t get_array_size() const {
        return array_size;
    }

  public:
    static VkImageInHandle
    transfer_src(const std::string& name, const uint32_t delay = 0, const bool optional = false);

  private:
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
    const vk::ImageLayout required_layout;
    const vk::ImageUsageFlags usage_flags;
    const vk::ShaderStageFlags stage_flags;

    uint32_t array_size = 1;
};

} // namespace merian
