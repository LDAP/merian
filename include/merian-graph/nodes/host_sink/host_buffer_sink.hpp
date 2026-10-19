#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_in.hpp"
#include "merian-graph/nodes/host_sink/host_sink.hpp"

namespace merian {

class HostBufferSink : public HostSink {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override;

    NodeStatusFlags properties(Properties& config) override;

  protected:
    bool capturable(const NodeIO& io) const override;

    HostCopy cmd_capture(const CommandBufferHandle& cmd,
                         const NodeIO& io,
                         const NodeProcessInfo& info) override;

  private:
    std::pair<BufferHandle, vk::DeviceSize> captured_range(const NodeIO& io) const;

    VkBufferInHandle con_src = VkBufferIn::create();

    vk::DeviceSize offset = 0;
    vk::DeviceSize size = 0;
    uint32_t element = 0;
};

} // namespace merian
