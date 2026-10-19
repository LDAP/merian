#pragma once

#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/nodes/host_sink/host_sink.hpp"

namespace merian {

class HostImageSink : public HostSink {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override;

    NodeStatusFlags on_connected(const NodeIOLayout& io_layout,
                                 const NodeIO& io,
                                 const NodeConnectionInfo& info,
                                 Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

  protected:
    HostCopy cmd_capture(const CommandBufferHandle& cmd,
                         const NodeIO& io,
                         const NodeProcessInfo& info) override;

  private:
    vk::Extent3D fitted(const vk::Extent3D& extent) const;

    VkImageInHandle con_src = VkImageIn::create();

    vk::Format format = vk::Format::eUndefined;
    vk::Extent2D fit{};
    vk::Format src_format = vk::Format::eUndefined;
    vk::Filter filter = vk::Filter::eLinear;
    bool blittable = false;
    ImageHandle conversion_image;
};

} // namespace merian
