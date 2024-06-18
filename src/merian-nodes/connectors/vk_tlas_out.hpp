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
    // No descriptor binding is created.
    VkTLASOut(const std::string& name);

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resoruce_index,
                    const uint32_t ring_size) override;

    AccelerationStructureHandle& resource(const GraphResourceHandle& resource) override;

    ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const vk::CommandBuffer& cmd,
                    GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  public:
    static VkTLASOutHandle create(const std::string& name);
};

} // namespace merian_nodes
