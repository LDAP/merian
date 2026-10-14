#pragma once

#include "merian-graph/resources/image_array_resource.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

// An image the producing node sets every iteration, or none.
class ExternalImageArrayResource : public ImageArrayResource {
  public:
    ExternalImageArrayResource(const SamplerHandle& sampler)
        : ImageArrayResource(1), sampler(sampler) {}

    const ImageHandle& get_image([[maybe_unused]] const uint32_t index = 0) const override {
        assert(index == 0);
        return texture ? texture->get_image() : Image::EMPTY;
    }

    const TextureHandle& get_texture([[maybe_unused]] const uint32_t index = 0) const override {
        assert(index == 0);
        return texture;
    }

    void set(const ImageViewHandle& view) {
        texture = view ? Texture::create(view, sampler) : nullptr;
    }

  private:
    const SamplerHandle sampler;
    TextureHandle texture;
};

} // namespace merian
