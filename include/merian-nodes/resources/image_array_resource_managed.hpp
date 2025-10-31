#pragma once

#include "merian-nodes/resources/image_array_resource.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

class ManagedImageArrayResource : public ImageArrayResource {
    friend class ManagedVkImageOut;

  public:
    ManagedImageArrayResource(uint32_t array_size,
                              const vk::PipelineStageFlags2 input_stage_flags,
                              const vk::AccessFlags2 input_access_flags)
        : ImageArrayResource(array_size, input_stage_flags, input_access_flags),
          images(array_size) {}

    // can be nullptr
    virtual const merian::ImageHandle& get_image(const uint32_t index) const override {
        assert(index < images.size());
        return images[index];
    }

    // can be nullptr
    virtual const merian::TextureHandle& get_texture(const uint32_t index) const override {
        if (!textures) {
            return merian::Texture::EMPTY;
        }

        assert(index < textures->size());
        return textures.value()[index];
    }

  private:
    std::vector<merian::ImageHandle> images;

    std::optional<std::vector<merian::TextureHandle>>
        textures; // has value if usage flags indicate use as view.
};

using ManagedImageArrayResourceHandle = std::shared_ptr<ManagedImageArrayResource>;

} // namespace merian
