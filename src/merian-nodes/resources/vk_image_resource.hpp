#pragma once

#include "graph/resource.hpp"

namespace merian_nodes {

class VkTextureResource : public GraphResource {
    friend class VkImageIn;
    friend class VkImageOut;

  public:
    VkTextureResource(const TextureHandle& tex,
                      const vk::PipelineStageFlags2& input_stage_flags,
                      const vk::AccessFlags2& input_access_flags)
        : tex(tex), input_stage_flags(input_stage_flags), input_access_flags(input_access_flags) {}

  private:
    const TextureHandle tex;

    // for barrier insertions
    vk::PipelineStageFlags2 current_stage_flags = vk::PipelineStageFlagBits2::eTopOfPipe;
    vk::AccessFlags2 current_access_flags{};

    bool needs_descriptor_update = true;
    bool last_used_as_output = false;

    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;
};

using VkTextureResourceHandle = std::shared_ptr<VkTextureResource>;

} // namespace merian_nodes
