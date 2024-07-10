#pragma once

#include "vk_tlas_out.hpp"

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/resources/tlas_resource.hpp"

namespace merian_nodes {

class VkTLASIn;
using VkTLASInHandle = std::shared_ptr<VkTLASIn>;

// Input a TLAS.
class VkTLASIn : public TypedInputConnector<VkTLASOutHandle, const AccelerationStructureHandle&> {
    friend class VkTLASOut;

  public:
    VkTLASIn(const std::string& name,
             const vk::ShaderStageFlags stage_flags,
             const vk::PipelineStageFlags2 pipeline_stages);

    void on_connect_output(const OutputConnectorHandle& output) override;

    std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    void get_descriptor_update(const uint32_t binding,
                               const GraphResourceHandle& resource,
                               DescriptorSetUpdate& update,
                               const ResourceAllocatorHandle& allocator) override;

    const AccelerationStructureHandle& resource(const GraphResourceHandle& resource) override;

  public:
    // Creates an output that has to set the TLAS and can it read in a shader.
    static VkTLASInHandle compute_read(const std::string& name);

  private:
    const vk::ShaderStageFlags stage_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
};

} // namespace merian_nodes
