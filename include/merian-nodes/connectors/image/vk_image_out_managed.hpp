#pragma once

#include "vk_image_out.hpp"

#include "merian-nodes/resources/image_array_resource.hpp"

namespace merian_nodes {

class ManagedVkImageOut;
using ManagedVkImageOutHandle = std::shared_ptr<ManagedVkImageOut>;

// Output a Vulkan image that is allocated and managed by the graph.
// Note that it only supplies a descriptor if stage_flags contains at least one bit.
class ManagedVkImageOut : public VkImageOut, public AccessibleConnector<const ImageArrayResource&> {
  public:
    ManagedVkImageOut(const std::string& name,
                      const vk::AccessFlags2& access_flags,
                      const vk::PipelineStageFlags2& pipeline_stages,
                      const vk::ImageLayout& required_layout,
                      const vk::ShaderStageFlags& stage_flags,
                      const vk::ArrayProxy<vk::ImageCreateInfo>& create_info,
                      const bool persistent = false);

    std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    void get_descriptor_update(const uint32_t binding,
                               const GraphResourceHandle& resource,
                               const DescriptorSetHandle& update,
                               const ResourceAllocatorHandle& allocator) override;

    const ImageArrayResource& resource(const GraphResourceHandle& resource) override;

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

    GraphResourceHandle
    create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                    const ResourceAllocatorHandle& allocator,
                    const ResourceAllocatorHandle& aliasing_allocator,
                    const uint32_t resource_index,
                    const uint32_t ring_size) override;

    virtual std::optional<vk::ImageCreateInfo>
    get_create_info(const uint32_t index = 0) const override;

  public:
    static ManagedVkImageOutHandle create(const std::string& name,
                                          const vk::AccessFlags2& access_flags,
                                          const vk::PipelineStageFlags2& pipeline_stages,
                                          const vk::ImageLayout& required_layout,
                                          const vk::ShaderStageFlags& stage_flags,
                                          const vk::ArrayProxy<vk::ImageCreateInfo>& create_info,
                                          const bool persistent = false);

    static ManagedVkImageOutHandle compute_write(const std::string& name,
                                                 const vk::Format format,
                                                 const vk::Extent3D extent,
                                                 const bool persistent = false);

    // uses the eGeneral Layout
    static ManagedVkImageOutHandle compute_fragment_write(const std::string& name,
                                                          const vk::Format format,
                                                          const vk::Extent3D extent,
                                                          const bool persistent = false);

    static ManagedVkImageOutHandle fragment_write(const std::string& name,
                                                  const vk::Format format,
                                                  const vk::Extent3D extent,
                                                  const bool persistent = false);

    static ManagedVkImageOutHandle color_attachment(const std::string& name,
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

  public:
    const vk::AccessFlags2 access_flags;
    const vk::PipelineStageFlags2 pipeline_stages;
    const vk::ImageLayout required_layout;
    const vk::ShaderStageFlags stage_flags;

  private:
    const std::vector<vk::ImageCreateInfo> create_infos;
};

} // namespace merian_nodes
