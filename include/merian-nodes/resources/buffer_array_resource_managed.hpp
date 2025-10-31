#pragma once

#include "merian-nodes/resources/buffer_array_resource.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

class ManagedBufferArrayResource : public BufferArrayResource {
    friend class ManagedVkBufferOut;

  public:
    ManagedBufferArrayResource(const uint32_t array_size,
                               const vk::BufferUsageFlags buffer_usage_flags,
                               const vk::PipelineStageFlags2 input_stage_flags,
                               const vk::AccessFlags2 input_access_flags)
        : BufferArrayResource(
              array_size, buffer_usage_flags, input_stage_flags, input_access_flags),
          buffers(array_size) {
    }

    // can be nullptr
    const merian::BufferHandle& get_buffer(const uint32_t index) const override {
        assert(index < buffers.size());
        return buffers[index];
    }

  private:
    std::vector<merian::BufferHandle> buffers;
};

} // namespace merian
