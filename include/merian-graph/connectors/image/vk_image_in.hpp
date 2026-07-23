#pragma once

#include "vk_image_out.hpp"

#include "merian-graph/graph/connector_input.hpp"
#include "merian-graph/resources/image_array_resource.hpp"

namespace merian {

class VkImageIn;
using VkImageInHandle = std::shared_ptr<VkImageIn>;

// Receives an image (array). Declare how the node accesses it via ConnectorAccess in the
// descriptor; the graph synchronizes.
class VkImageIn : public InputConnector,
                  public OutputAccessibleInputConnector<VkImageOutHandle>,
                  public AccessibleConnector<const ImageArrayResource&> {
  public:
    VkImageIn() = default;

    void on_connect_output(const OutputConnectorHandle& output) override;

    // A delayed input reads a ring slot before its producer ever transitioned it.
    ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const CommandBufferHandle& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    virtual const ImageArrayResource& resource(const GraphResourceHandle& resource) override;

    uint32_t get_array_size() const {
        return array_size;
    }

  public:
    static VkImageInHandle create();

  private:
    uint32_t array_size = 1;
};

} // namespace merian
