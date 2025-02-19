#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class ImageArrayResource : public GraphResource {
    friend class VkImageOut;
    friend class VkImageIn;

  public:
    ImageArrayResource(std::vector<merian::ImageHandle>& images,
                         const merian::ImageHandle& dummy_image,
                         const vk::PipelineStageFlags2 input_stage_flags,
                         const vk::AccessFlags2 input_access_flags,
                         const vk::ImageLayout first_input_layout)
        : input_stage_flags(input_stage_flags), input_access_flags(input_access_flags),
          images(images), dummy_image(dummy_image), first_input_layout(first_input_layout) {

        for (uint32_t i = 0; i < images.size(); i++) {
            current_updates.push_back(i);
        }
    }

    // Automatically inserts barrier, can be nullptr to reset to dummy texture.
    void set(const uint32_t index,
             const merian::ImageHandle& image,
             const merian::CommandBufferHandle& cmd,
             const vk::AccessFlags2 prior_access_flags,
             const vk::PipelineStageFlags2 prior_pipeline_stages) {
        assert(index < textures.size());

        if (images[index] != image) {
            images[index] = image;
            current_updates.push_back(index);
        }

        if (image && first_input_layout != vk::ImageLayout::eUndefined &&
            (prior_access_flags || image->get_current_layout() != first_input_layout)) {
            const vk::ImageMemoryBarrier2 img_bar = image->barrier2(
                first_input_layout, prior_access_flags, input_access_flags, prior_pipeline_stages,
                input_stage_flags);

            cmd->barrier(img_bar);
        }
    }

    const merian::ImageHandle& get(const uint32_t index) const {
        assert(index < textures.size());
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
                props.st_begin_child(std::to_string(i), fmt::format("Image {:04d}", i))) {
                images[i]->properties(props);
                props.st_end_child();
            }
        }
    }

  public:
    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;

  private:
    // the updates to "textures" are recorded here.
    std::vector<uint32_t> current_updates;
    // then flushed to here to wait for the graph to apply descriptor updates.
    std::vector<uint32_t> pending_updates;

    std::vector<merian::ImageHandle>& images;

    const merian::ImageHandle dummy_image;
    const vk::ImageLayout first_input_layout;
};

} // namespace merian_nodes
