#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class TextureArrayResource : public GraphResource {
    friend class VkTextureArrayOut;
    friend class VkTextureArrayIn;

  public:
    TextureArrayResource(std::vector<merian::TextureHandle>& textures,
                         const uint32_t ring_size,
                         const merian::TextureHandle& dummy_texture,
                         const vk::PipelineStageFlags2 input_stage_flags,
                         const vk::AccessFlags2 input_access_flags,
                         const vk::ImageLayout first_input_layout)
        : input_stage_flags(input_stage_flags), input_access_flags(input_access_flags),
          textures(textures), dummy_texture(dummy_texture), first_input_layout(first_input_layout) {
        in_flight_textures.assign(ring_size, std::vector<merian::TextureHandle>(textures.size()));

        for (uint32_t i = 0; i < textures.size(); i++) {
            current_updates.push_back(i);
        }
    }

    // Automatically inserts barrier, can be nullptr to reset to dummy texture.
    void set(const uint32_t index,
             const merian::TextureHandle& tex,
             const vk::CommandBuffer cmd,
             const vk::AccessFlags2 prior_access_flags,
             const vk::PipelineStageFlags2 prior_pipeline_stages) {
        assert(index < textures.size());

        if (textures[index] != tex) {
            textures[index] = tex;
            current_updates.push_back(index);
        }

        if (tex &&
            (prior_access_flags || tex->get_image()->get_current_layout() != first_input_layout)) {
            const vk::ImageMemoryBarrier2 img_bar = tex->get_image()->barrier2(
                first_input_layout, prior_access_flags, input_access_flags, prior_pipeline_stages,
                input_stage_flags);

            cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, {}, img_bar});
        }
    }

    const merian::TextureHandle& get(const uint32_t index) const {
        assert(index < textures.size());
        return textures[index];
    }

    void properties(merian::Properties& props) override {
        props.output_text(
            fmt::format("Array size: {}\nCurrent updates: {}\nPending updates: {}\nInput access "
                        "flags: {}\nInput pipeline stages: {}\nInput first layout: {}",
                        textures.size(), current_updates.size(), pending_updates.size(),
                        vk::to_string(input_access_flags), vk::to_string(input_stage_flags),
                        vk::to_string(first_input_layout)));
        for (uint32_t i = 0; i < textures.size(); i++) {
            if (textures[i] &&
                props.st_begin_child(std::to_string(i), fmt::format("Texture {:04d}", i))) {
                textures[i]->properties(props);
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

    std::vector<merian::TextureHandle>& textures;
    // on_post_process copy here to keep alive
    std::vector<std::vector<merian::TextureHandle>> in_flight_textures;

    const merian::TextureHandle dummy_texture;
    const vk::ImageLayout first_input_layout;
};

} // namespace merian_nodes
