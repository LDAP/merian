#ifndef MANAGED_VK_IMAGE_OUT_HPP
#define MANAGED_VK_IMAGE_OUT_HPP
#include "vk_image_out.hpp"

namespace merian_nodes {
class ManagedVkImageOut;
using ManagedVkImageOutHandle = std::shared_ptr<ManagedVkImageOut>;

class ManagedVkImageOut : public VkImageOut {
public:
  ManagedVkImageOut(const std::string& name,
                    const vk::AccessFlags2& access_flags,
                    const vk::PipelineStageFlags2& pipeline_stages,
                    const vk::ImageLayout& required_layout,
                    const vk::ShaderStageFlags& stage_flags,
                    const vk::ImageCreateInfo& create_info,
                    const bool persistent = false,
                    const uint32_t image_count = 1
                    );

  virtual GraphResourceHandle
  create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                  const ResourceAllocatorHandle& allocator,
                  const ResourceAllocatorHandle& aliasing_allocator,
                  const uint32_t resource_index,
                  const uint32_t ring_size) override;

  virtual ConnectorStatusFlags
  on_pre_process(GraphRun& run,
               const CommandBufferHandle& cmd,
               const GraphResourceHandle& resource,
               const NodeHandle& node,
               std::vector<vk::ImageMemoryBarrier2>& image_barriers,
               std::vector<vk::BufferMemoryBarrier2>& buffer_barriers) override;

  static ManagedVkImageOutHandle compute_write(const std::string& name,
                                               const vk::Format format,
                                               const vk::Extent3D extent,
                                               const bool persistent = false);

  static ManagedVkImageOutHandle compute_fragment_write(const std::string& name,
                                                        const vk::Format format,
                                                        const vk::Extent3D extent,
                                                        const bool persistent = false);

  static ManagedVkImageOutHandle compute_write(const std::string& name,
                                               const vk::Format format,
                                               const uint32_t width,
                                               const uint32_t height,
                                               const uint32_t depth = 1,
                                               const bool persistent = false);

  static ManagedVkImageOutHandle compute_fragment_write(const std::string& name,
                                                        const vk::Format format,
                                                        const uint32_t width,
                                                        const uint32_t height,
                                                        const uint32_t depth = 1,
                                                        const bool persistent = false);

  static ManagedVkImageOutHandle
  compute_read_write_transfer_dst(const std::string& name,
                                  const vk::Format format,
                                  const vk::Extent3D extent,
                                  const vk::ImageLayout layout = vk::ImageLayout::eGeneral,
                                  const bool persistent = false);

  static ManagedVkImageOutHandle
  compute_read_write_transfer_dst(const std::string& name,
                                  const vk::Format format,
                                  const uint32_t width,
                                  const uint32_t height,
                                  const uint32_t depth = 1,
                                  const vk::ImageLayout layout = vk::ImageLayout::eGeneral,
                                  const bool persistent = false);

  static ManagedVkImageOutHandle compute_read_write(const std::string& name,
                                                    const vk::Format format,
                                                    const vk::Extent3D extent,
                                                    const bool persistent = false);

  static ManagedVkImageOutHandle transfer_write(const std::string& name,
                                                const vk::Format format,
                                                const vk::Extent3D extent,
                                                const bool persistent = false);

  static ManagedVkImageOutHandle transfer_write(const std::string& name,
                                                const vk::Format format,
                                                const uint32_t width,
                                                const uint32_t height,
                                                const uint32_t depth = 1,
                                                const bool persistent = false);
};
} // namespace merian-nodes


#endif //MANAGED_VK_IMAGE_OUT_HPP
