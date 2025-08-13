#pragma once

#include "merian-nodes/graph/resource.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

/**
 * @brief      This class describes an image array resource.
 *
 * Note: textures must exist if all usages flags combined (output + all inputs) suggest use as view.
 */
class ImageArrayResource : public GraphResource {
    friend class ManagedVkImageOut;
    friend class UnmanagedVkImageOut;
    friend class VkImageIn;
    friend class VkSampledImageIn;
    friend class VkStorageImageIn;

  public:
    ImageArrayResource(uint32_t array_size,
                       const vk::PipelineStageFlags2 input_stage_flags,
                       const vk::AccessFlags2 input_access_flags)
        : array_size(array_size), input_stage_flags(input_stage_flags),
          input_access_flags(input_access_flags) {

        for (uint32_t i = 0; i < array_size; i++) {
            current_updates.push_back(i);
        }
    }

    // can be nullptr
    virtual const merian::ImageHandle& get_image(const uint32_t index = 0) const = 0;

    // can be nullptr
    virtual const merian::TextureHandle& get_texture(const uint32_t index = 0) const = 0;

    uint32_t get_array_size() const {
        return array_size;
    }

    vk::PipelineStageFlags2 get_input_stage_flags() const {
        return input_stage_flags;
    }

    vk::AccessFlags2 get_input_access_flags() const {
        return input_access_flags;
    }

    void properties(merian::Properties& props) override {
        props.output_text(fmt::format("Array size: {}\nCurrent updates: {}\nPending updates: "
                                      "{}\nInput access flags: {}\nInput pipeline stages: {}",
                                      array_size, current_updates.size(), pending_updates.size(),
                                      vk::to_string(input_access_flags),
                                      vk::to_string(input_stage_flags)));

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

  protected:
    void queue_descriptor_update(const uint32_t array_index) {
        current_updates.emplace_back(array_index);
    }

  private:
    uint32_t array_size;

    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;

    // for barrier insertions
    vk::PipelineStageFlags2 current_stage_flags = vk::PipelineStageFlagBits2::eTopOfPipe;
    vk::AccessFlags2 current_access_flags{};

    // the updates to "textures" are recorded here.
    std::vector<uint32_t> current_updates;
    // then flushed to here to wait for the graph to apply descriptor updates.
    std::vector<uint32_t> pending_updates;

    bool last_used_as_output = true;
};

using ImageArrayResourceHandle = std::shared_ptr<ImageArrayResource>;

} // namespace merian_nodes
