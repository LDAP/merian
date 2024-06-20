#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/resources/tlas_resource.hpp"

namespace merian_nodes {

class VkTLASOut;
using VkTLASOutHandle = std::shared_ptr<VkTLASOut>;

// Output a TLAS.
//
// Note that this connector does also persists the tlas accross graph rebuilds.
//
// The output keeps the tlas alive for all in-flight iterations.
class VkTLASOut : public TypedOutputConnector<AccelerationStructureHandle&> {
    friend class VkTLASIn;

  public:
    // A descriptor binding is only created if stage_flags is not empty.
    VkTLASOut(const std::string& name, const vk::ShaderStageFlags stage_flags);

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
    static VkTLASOutHandle compute_read(const std::string& name);

    // Creates an output that has to set the TLAS (but a descriptor binding is not created).
    static VkTLASOutHandle create(const std::string& name);

  private:
    const vk::ShaderStageFlags stage_flags;
};

} // namespace merian_nodes
