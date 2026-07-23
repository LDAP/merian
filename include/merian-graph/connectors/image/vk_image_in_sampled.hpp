#pragma once

#include "vk_image_in.hpp"

namespace merian {

class VkSampledImageIn;
using VkSampledImageInHandle = std::shared_ptr<VkSampledImageIn>;

class VkSampledImageIn : public VkImageIn {
  public:
    VkSampledImageIn(const std::optional<SamplerHandle>& overwrite_sampler = std::nullopt);

    bool shader_bindable() const override {
        return true;
    }

    void bind(ShaderCursor& cursor,
              const GraphResourceHandle& resource,
              const ResourceAllocatorHandle& allocator,
              const ConnectorAccess& access) override;

  public:
    static VkSampledImageInHandle
    create(const std::optional<SamplerHandle>& overwrite_sampler = std::nullopt);

  private:
    const std::optional<SamplerHandle> overwrite_sampler;
};

} // namespace merian
