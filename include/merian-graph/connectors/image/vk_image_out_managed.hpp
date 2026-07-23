#pragma once

#include "vk_image_out.hpp"

#include "merian-graph/resources/image_array_resource.hpp"

namespace merian {

class ManagedVkImageOut;
using ManagedVkImageOutHandle = std::shared_ptr<ManagedVkImageOut>;

// Output a Vulkan image that is allocated and managed by the graph. Usage flags are the union
// of the declared ConnectorAccess of this port and all connected inputs.
class ManagedVkImageOut : public VkImageOut, public AccessibleConnector<const ImageArrayResource&> {
  public:
    ManagedVkImageOut(const vk::ArrayProxy<vk::ImageCreateInfo>& create_info,
                      const bool persistent = false);

    bool shader_bindable() const override {
        return true;
    }

    void bind(ShaderCursor& cursor,
              const GraphResourceHandle& resource,
              const ResourceAllocatorHandle& allocator,
              const ConnectorAccess& access) override;

    const ImageArrayResource& resource(const GraphResourceHandle& resource) override;

    ConnectorStatusFlags
    on_pre_process(GraphRun& run,
                   const CommandBufferHandle& cmd,
                   const GraphResourceHandle& resource,
                   const NodeHandle& node,
                   std::vector<vk::ImageMemoryBarrier2>& image_barriers,
                   std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ConnectorAccess& combined_access,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resource_index,
                    const uint32_t ring_size) override;

    virtual std::optional<vk::ImageCreateInfo>
    get_create_info(const uint32_t index = 0) const override;

  public:
    static ManagedVkImageOutHandle
    create(const vk::Format format, const vk::Extent3D extent, const bool persistent = false);

    static ManagedVkImageOutHandle create(const vk::Format format,
                                          const uint32_t width,
                                          const uint32_t height,
                                          const uint32_t depth = 1,
                                          const bool persistent = false);

    static ManagedVkImageOutHandle create(const vk::ArrayProxy<vk::ImageCreateInfo>& create_info,
                                          const bool persistent = false);

  private:
    const std::vector<vk::ImageCreateInfo> create_infos;
};

} // namespace merian
