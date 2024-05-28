#pragma once

#include "merian-nodes/graph/resource.hpp"

namespace merian_nodes {

class VkBufferResource : public GraphResource {
    friend class VkBufferIn;
    friend class VkBufferOut;

  public:
    VkBufferResource(const BufferHandle& buffer,
                     const vk::PipelineStageFlags2& input_stage_flags,
                     const vk::AccessFlags2& input_access_flags)
        : buffer(buffer), input_stage_flags(input_stage_flags),
          input_access_flags(input_access_flags) {}

  private:
    const BufferHandle buffer;

    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;

    bool needs_descriptor_update = true;
};

using VkBufferResourceHandle = std::shared_ptr<VkBufferResource>;

} // namespace merian_nodes
