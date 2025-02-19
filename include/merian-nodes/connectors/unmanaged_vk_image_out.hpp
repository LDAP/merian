#ifndef UNMANAGED_VK_IMAGE_OUT_HPP
#define UNMANAGED_VK_IMAGE_OUT_HPP

#include "vk_image_out.hpp"

namespace merian_nodes {

class UnmanagedVkImageOut;
using UnmanagedVkImageOutHandle = std::shared_ptr<UnmanagedVkImageOut>;

class UnmanagedVkImageOut : public VkImageOut {
public:
  UnmanagedVkImageOut(const std::string& name, const uint32_t array_size);

  virtual GraphResourceHandle
  create_resource(const std::vector<std::tuple<NodeHandle, InputConnectorHandle>>& inputs,
                const ResourceAllocatorHandle& allocator,
                const ResourceAllocatorHandle& aliasing_allocator,
                const uint32_t resource_index,
                const uint32_t ring_size) override;

  static UnmanagedVkImageOutHandle create(const std::string& name, const uint32_t array_size);

private:
};

} // namespace merain_nodes



#endif //UNMANAGED_VK_IMAGE_OUT_HPP
