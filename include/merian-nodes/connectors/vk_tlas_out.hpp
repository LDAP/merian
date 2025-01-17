#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/resources/tlas_resource.hpp"

namespace merian_nodes {

class VkTLASOut;
using VkTLASOutHandle = std::shared_ptr<VkTLASOut>;

// Output a TLAS.
//
// The output keeps the tlas alive for all in-flight iterations.
//
// A barrier is inserted for the TLAS after the node has processed.
// Note, that if the node with this connector wants to use the TLAS (by providing stage flags), it
// must synchronize it manually. The reason for this is, that the TLAS can only be build in
// Node::process and if the node wants to use the TLAS itself there is no way to insert a barrier at
// the correct place and the node must insert the barrier itself.
//
// Note, that you are responsible to insert read->build barriers manually since the connector is
// unable to detect if a TLAS is reused or not. You can get the read stages using
// io[connector].read_pipeline_stages.
class VkTLASOut : public TypedOutputConnector<TLASResource&> {
    friend class VkTLASIn;

  public:
    // A descriptor binding is only created if stage_flags is not empty.
    VkTLASOut(const std::string& name);

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resoruce_index,
                    const uint32_t ring_size) override;

    TLASResource& resource(const GraphResourceHandle& resource) override;

    ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const CommandBufferHandle& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    ConnectorStatusFlags
    on_post_process(GraphRun& run,
                    const CommandBufferHandle& cmd,
                    const GraphResourceHandle& resource,
                    const NodeHandle& node,
                    std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                    std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  public:
    // Creates an output that has to set the TLAS.
    static VkTLASOutHandle create(const std::string& name);
};

} // namespace merian_nodes
