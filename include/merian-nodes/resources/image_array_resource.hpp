#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class ImageArrayResource : public GraphResource {
    friend class ManagedVkImageIn;
    friend class ManagedVkImageOut;

  public:
    ImageArrayResource(std::vector<merian::ImageHandle>& images,
                           const vk::PipelineStageFlags2& input_stage_flags,
                           const vk::AccessFlags2& input_access_flags)
        : images(images), textures(images.size()), input_stage_flags(input_stage_flags),
          input_access_flags(input_access_flags), first_input_layout(vk::ImageLayout::eUndefined) {
            for (uint32_t i = 0; i < images.size(); i++) {
              current_updates.push_back(i);
            }
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
    std::vector<std::optional<merian::TextureHandle>> textures;

    // for barrier insertions
    vk::PipelineStageFlags2 current_stage_flags = vk::PipelineStageFlagBits2::eTopOfPipe;
    vk::AccessFlags2 current_access_flags{};

    // the updates to "textures" are recorded here.
    std::vector<uint32_t> current_updates;
    // then flushed to here to wait for the graph to apply descriptor updates.
    std::vector<uint32_t> pending_updates;

    bool last_used_as_output = true;

    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;
    const vk::ImageLayout first_input_layout; // TODO use this and init it from the constructor

};

using ImageArrayResourceHandle = std::shared_ptr<ImageArrayResource>;

} // namespace merian_nodes
