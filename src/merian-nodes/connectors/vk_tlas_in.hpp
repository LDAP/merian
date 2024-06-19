#pragma once

#include "merian-nodes/graph/connector_input.hpp"
#include "merian-nodes/resources/tlas_resource.hpp"

namespace merian_nodes {

class VkTLASIn;
using VkTLASInHandle = std::shared_ptr<VkTLASIn>;

// Input a TLAS.
class VkTLASIn : public TypedInputConnector<VkTLASOutHandle, const AccelerationStructureHandle&> {
    friend class VkTLASOut;

  public:
    VkTLASIn(const std::string& name, const vk::ShaderStageFlags stage_flags);

    std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    void get_descriptor_update(const uint32_t binding,
                               GraphResourceHandle& resource,
                               DescriptorSetUpdate& update) override;

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resoruce_index,
                    const uint32_t ring_size) override;

    AccelerationStructureHandle& resource(const GraphResourceHandle& resource) override;

    ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const vk::CommandBuffer& cmd,
                   GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const vk::CommandBuffer& cmd,
                    GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  public:
    // Creates an output that has to set the TLAS and can it read in a shader.
    static VkTLASInHandle compute_read(const std::string& name);

  private:
    const vk::ShaderStageFlags stage_flags;
};

} // namespace merian_nodes
