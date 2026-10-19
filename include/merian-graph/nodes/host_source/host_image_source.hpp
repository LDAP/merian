#pragma once

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/nodes/host_source/host_source.hpp"

namespace merian {

class HostImageSource : public HostSource {
  public:
    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    // Reconnects if format or extent change; fill writes tightly packed texels.
    void upload(const vk::Format format,
                const vk::Extent3D extent,
                const std::function<void(void* texels)>& fill);

  protected:
    void cmd_copy(const CommandBufferHandle& cmd,
                  const NodeIO& io,
                  const BufferHandle& staging_buffer,
                  const vk::DeviceSize staging_offset) override;

  private:
    ManagedVkImageOutHandle con_out;

    vk::Format format = vk::Format::eUndefined;
    vk::Extent3D extent{};
};

} // namespace merian
