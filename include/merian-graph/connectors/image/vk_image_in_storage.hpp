#pragma once

#include "vk_image_in.hpp"

namespace merian {

class VkStorageImageIn;
using VkStorageImageInHandle = std::shared_ptr<VkStorageImageIn>;

class VkStorageImageIn : public VkImageIn {
  public:
    VkStorageImageIn() = default;

    bool shader_bindable() const override {
        return true;
    }

    void bind(ShaderCursor& cursor,
              const GraphResourceHandle& resource,
              const ResourceAllocatorHandle& allocator,
              const ConnectorAccess& access) override;

  public:
    static VkStorageImageInHandle create();
};

} // namespace merian
