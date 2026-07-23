#pragma once

#include "merian-graph/resources/buffer_array_resource.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

class ManagedBufferArrayResource : public BufferArrayResource {
    friend class ManagedVkBufferOut;

  public:
    ManagedBufferArrayResource(const uint32_t array_size)
        : BufferArrayResource(array_size), buffers(array_size) {}

    // can be nullptr
    const merian::BufferHandle& get_buffer(const uint32_t index) const override {
        assert(index < buffers.size());
        return buffers[index];
    }

  private:
    std::vector<merian::BufferHandle> buffers;
};

} // namespace merian
