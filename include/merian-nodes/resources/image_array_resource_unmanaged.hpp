#pragma once

#include "merian-nodes/resources/image_array_resource.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

class UnmanagedImageArrayResource : public ImageArrayResource {
    friend class UnmanagedVkImageOut;

  public:
    UnmanagedImageArrayResource(
        std::vector<merian::ImageHandle>& images,
        std::optional<std::vector<merian::TextureHandle>>& textures,
        const vk::ImageUsageFlags image_usage_flags,
        const vk::PipelineStageFlags2 input_stage_flags,
        const vk::AccessFlags2 input_access_flags,
        const vk::ImageLayout first_input_layout = vk::ImageLayout::eUndefined)
        : ImageArrayResource(images.size(), input_stage_flags, input_access_flags), images(images),
          textures(textures), image_usage_flags(image_usage_flags),
          first_input_layout(first_input_layout) {}

    void set(const uint32_t index,
             const merian::TextureHandle& tex,
             const merian::CommandBufferHandle& cmd,
             const vk::AccessFlags2 prior_access_flags,
             const vk::PipelineStageFlags2 prior_pipeline_stages) {
        assert(index < images.size());
        assert((tex->get_image()->get_usage_flags() & image_usage_flags) ==
               image_usage_flags); // node must supply an image that has all usage flags from inputs
                                   // and output set.
        assert(tex);

        if (textures.has_value()) {
            if (textures.value()[index] != tex) {
                textures.value()[index] = tex;
                images[index] = tex->get_image();
                queue_descriptor_update(index);
            }
        } else {
            if (images[index] != tex->get_image()) {
                images[index] = tex->get_image();
                queue_descriptor_update(index);
            }
        }

        if (first_input_layout != vk::ImageLayout::eUndefined &&
            (prior_access_flags || images[index]->get_current_layout() != first_input_layout)) {
            const vk::ImageMemoryBarrier2 img_bar = images[index]->barrier2(
                first_input_layout, prior_access_flags, get_input_access_flags(),
                prior_pipeline_stages, get_input_stage_flags());

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
        assert((image->get_usage_flags() & image_usage_flags) ==
               image_usage_flags); // node must supply an image that has all usage flags from inputs
                                   // and output set.

        if (images[index] != image) {
            images[index] = image;
            if (textures.has_value()) {
                textures.value()[index] = image ? allocator->create_texture(image) : nullptr;
            }
            queue_descriptor_update(index);
        }

        if (first_input_layout != vk::ImageLayout::eUndefined &&
            (prior_access_flags || images[index]->get_current_layout() != first_input_layout)) {
            const vk::ImageMemoryBarrier2 img_bar = images[index]->barrier2(
                first_input_layout, prior_access_flags, get_input_access_flags(),
                prior_pipeline_stages, get_input_stage_flags());
            cmd->barrier(img_bar);
        }
    }

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

    void properties(merian::Properties& props) override {
        ImageArrayResource::properties(props);

        props.output_text(fmt::format("Input first layout: {}", vk::to_string(first_input_layout)));
    }

  private:
    std::vector<merian::ImageHandle>& images;

    std::optional<std::vector<merian::TextureHandle>>&
        textures; // has value if usage flags indicate use as view.

    // combined image usage flags of all inputs and outputs
    const vk::ImageUsageFlags image_usage_flags;

    const vk::ImageLayout first_input_layout;
};

using UnmanagedImageArrayResourceHandle = std::shared_ptr<UnmanagedImageArrayResource>;

} // namespace merian
