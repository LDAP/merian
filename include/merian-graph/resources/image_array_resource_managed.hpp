#pragma once

#include "merian-graph/resources/image_array_resource.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

class ManagedImageArrayResource : public ImageArrayResource {
    friend class ManagedVkImageOut;

  public:
    ManagedImageArrayResource(uint32_t array_size)
        : ImageArrayResource(array_size), images(array_size) {}

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
