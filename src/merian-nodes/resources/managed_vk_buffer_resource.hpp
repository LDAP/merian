#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class ManagedVkBufferResource : public GraphResource {
    friend class ManagedVkBufferIn;
    friend class ManagedVkBufferOut;

  public:
    ManagedVkBufferResource(const merian::BufferHandle& buffer,
                            const vk::PipelineStageFlags2& input_stage_flags,
                            const vk::AccessFlags2& input_access_flags)
        : buffer(buffer), input_stage_flags(input_stage_flags),
          input_access_flags(input_access_flags) {}

    void properties(merian::Properties& props) override {
        buffer->properties(props);
    }

  private:
    const merian::BufferHandle buffer;

    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;

    bool needs_descriptor_update = true;
};

using VkBufferResourceHandle = std::shared_ptr<ManagedVkBufferResource>;

} // namespace merian_nodes
