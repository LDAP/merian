#pragma once

#include "vk_image_in.hpp"

namespace merian {

class VkStorageImageIn;
using VkStorageImageInHandle = std::shared_ptr<VkStorageImageIn>;

class VkStorageImageIn : public VkImageIn {
  public:
    VkStorageImageIn(const std::string& name,
                     const vk::AccessFlags2 access_flags,
                     const vk::PipelineStageFlags2 pipeline_stages,
                     const vk::ImageUsageFlags usage_flags,
                     const vk::ShaderStageFlags stage_flags,
                     const uint32_t delay = 0,
                     const bool optional = false,
                     const vk::ImageLayout required_layout = vk::ImageLayout::eGeneral);

    virtual std::optional<vk::DescriptorSetLayoutBinding> get_descriptor_info() const override;

    // For a optional input, resource can be nullptr here to signalize that no output was connected.
    // Provide a dummy binding in this case so that descriptor sets do not need to change.
    virtual void get_descriptor_update(const uint32_t binding,
                                       const GraphResourceHandle& resource,
                                       const DescriptorSetHandle& update,
                                       const ResourceAllocatorHandle& allocator) override;

  public:
    static VkStorageImageInHandle
    compute_read(const std::string& name, const uint32_t delay = 0, const bool optional = false);
};

} // namespace merian
