#pragma once

#include "merian-graph/graph/resource.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

/**
 * @brief      This class describes an image array resource.
 *
 * Note: textures must exist if all usages flags combined (output + all inputs) suggest use as view.
 */
class ImageArrayResource : public GraphResource {
    friend class ManagedVkImageOut;

    friend class VkImageIn;
    friend class VkSampledImageIn;
    friend class VkStorageImageIn;

  public:
    ImageArrayResource(uint32_t array_size) : array_size(array_size) {}

    // can be nullptr
    virtual const merian::ImageHandle& get_image(const uint32_t index = 0) const = 0;

    // can be nullptr
    virtual const merian::TextureHandle& get_texture(const uint32_t index = 0) const = 0;

    uint32_t get_array_size() const {
        return array_size;
    }

    void properties(merian::Properties& props) override {
        props.output_text(fmt::format("Array size: {}", array_size));

        for (uint32_t i = 0; i < array_size; i++) {
            if (get_image(i) &&
                props.st_begin_child(std::to_string(i), fmt::format("Texture {:04d}", i))) {
                get_image(i)->properties(props);
                props.st_end_child();
            }
        }
    }

    merian::ImageHandle operator->() const {
        const merian::ImageHandle& image = get_image(0);
        assert(image);
        return image;
    }

    operator const merian::ImageHandle&() const {
        const merian::ImageHandle& image = get_image(0);
        assert(image);
        return image;
    }

    merian::Image& operator*() const {
        const merian::ImageHandle& image = get_image(0);
        assert(image);
        return *image;
    }

  private:
    uint32_t array_size;
};

using ImageArrayResourceHandle = std::shared_ptr<ImageArrayResource>;

} // namespace merian
