#pragma once

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/nodes/host_source/host_source.hpp"

namespace merian {

class HostBufferSource : public HostSource {
  public:
    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    // Reconnects if the size changes.
    void upload(const vk::DeviceSize size, const std::function<void(void* data)>& fill);

  protected:
    void cmd_copy(const CommandBufferHandle& cmd,
                  const NodeIO& io,
                  const BufferHandle& staging_buffer,
                  const vk::DeviceSize staging_offset) override;

  private:
    ManagedVkBufferOutHandle con_out;

    vk::DeviceSize size = 0;
};

} // namespace merian
