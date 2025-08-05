#pragma once

#include "merian-nodes/graph/resource.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian_nodes {

class ImageArrayResource : public GraphResource {
    friend class ManagedVkImageOut;
    friend class UnmanagedVkImageOut;
    friend class VkImageIn;
    friend class VkSampledImageIn;
    friend class VkStorageImageIn;

  public:
    ImageArrayResource(uint32_t array_size,
                       const vk::ImageUsageFlags& image_usage_flags,
                       const vk::PipelineStageFlags2& input_stage_flags,
                       const vk::AccessFlags2& input_access_flags,
                       const vk::ImageLayout first_input_layout = vk::ImageLayout::eUndefined)
        : images(array_size), image_usage_flags(image_usage_flags),
          input_stage_flags(input_stage_flags), input_access_flags(input_access_flags),
          first_input_layout(first_input_layout) {

        if (merian::Image::valid_for_view(image_usage_flags)) {
            textures.emplace(array_size);
        }

        for (uint32_t i = 0; i < images.size(); i++) {
            current_updates.push_back(i);
        }
    }

    void set(const uint32_t index,
             const merian::TextureHandle& tex,
             const merian::CommandBufferHandle& cmd,
             const vk::AccessFlags2 prior_access_flags,
             const vk::PipelineStageFlags2 prior_pipeline_stages) {
        assert(index < images.size());
        assert((tex->get_image()->get_usage_flags() & image_usage_flags) == image_usage_flags);
        assert(tex);

        if (textures.has_value()) {
            if (textures.value()[index] != tex) {
                textures.value()[index] = tex;
                images[index] = tex->get_image();
                current_updates.emplace_back(index);
            }
        } else {
            if (images[index] != tex->get_image()) {
                images[index] = tex->get_image();
                current_updates.emplace_back(index);
            }
        }

        if (first_input_layout != vk::ImageLayout::eUndefined &&
            (prior_access_flags || images[index]->get_current_layout() != first_input_layout)) {
            const vk::ImageMemoryBarrier2 img_bar =
                images[index]->barrier2(first_input_layout, prior_access_flags, input_access_flags,
                                        prior_pipeline_stages, input_stage_flags);

            cmd->barrier(img_bar);
        }
    }

    void set(const uint32_t index,
             const merian::ImageHandle& image,
             const merian::CommandBufferHandle& cmd,
             const merian::ResourceAllocatorHandle& allocator,
             const vk::AccessFlags2 prior_access_flags,
             const vk::PipelineStageFlags2 prior_pipeline_stages) {
        assert(index < images.size());
        assert((image->get_usage_flags() & image_usage_flags) == image_usage_flags);

        if (images[index] != image) {
            images[index] = image;
            if (textures.has_value()) {
                textures.value()[index] = image ? allocator->createTexture(image) : nullptr;
            }
            current_updates.emplace_back(index);
        }

        if (first_input_layout != vk::ImageLayout::eUndefined &&
            (prior_access_flags || images[index]->get_current_layout() != first_input_layout)) {
            const vk::ImageMemoryBarrier2 img_bar =
                images[index]->barrier2(first_input_layout, prior_access_flags, input_access_flags,
                                        prior_pipeline_stages, input_stage_flags);
            cmd->barrier(img_bar);
        }
    }

    // const merian::TextureHandle& get_texture(const uint32_t index) const {
    //     assert(index < textures.size());
    //     assert(textures[index].has_value() && "the image cannot be used as view.");
    //     return textures[index].value();
    // }

    const merian::ImageHandle& get_image(const uint32_t index) const {
        assert(index < images.size());
        return images[index];
    }

    void properties(merian::Properties& props) override {
        props.output_text(
            fmt::format("Array size: {}\nCurrent updates: {}\nPending updates: {}\nInput access "
                        "flags: {}\nInput pipeline stages: {}\nInput first layout: {}",
                        images.size(), current_updates.size(), pending_updates.size(),
                        vk::to_string(input_access_flags), vk::to_string(input_stage_flags),
                        vk::to_string(first_input_layout)));
        for (uint32_t i = 0; i < images.size(); i++) {
            if (images[i] &&
                props.st_begin_child(std::to_string(i), fmt::format("Texture {:04d}", i))) {
                images[i]->properties(props);
                props.st_end_child();
            }
        }
    }

    merian::ImageHandle operator->() const {
        assert(!images.empty());
        return images[0];
    }

    operator merian::ImageHandle() const {
        assert(!images.empty());
        return images[0];
    }

  private:
    std::vector<merian::ImageHandle> images;

    std::optional<std::vector<merian::TextureHandle>>
        textures; // has value if usage flags indicate use as view.

    // for barrier insertions
    vk::PipelineStageFlags2 current_stage_flags = vk::PipelineStageFlagBits2::eTopOfPipe;
    vk::AccessFlags2 current_access_flags{};

    // the updates to "textures" are recorded here.
    std::vector<uint32_t> current_updates;
    // then flushed to here to wait for the graph to apply descriptor updates.
    std::vector<uint32_t> pending_updates;

    bool last_used_as_output = true;

    // combined image usage flags of all inputs and outputs
    const vk::ImageUsageFlags image_usage_flags;
    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;
    const vk::ImageLayout first_input_layout;
};

using ImageArrayResourceHandle = std::shared_ptr<ImageArrayResource>;

} // namespace merian_nodes
