#pragma once

#include "merian-graph/graph/resource.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

class BufferArrayResource : public GraphResource {
    friend class VkBufferIn;
    friend class ManagedVkBufferOut;

  public:
    BufferArrayResource(const uint32_t array_size) : array_size(array_size) {}

    // can be nullptr
    virtual const merian::BufferHandle& get_buffer(const uint32_t index) const = 0;

    void properties(merian::Properties& props) override {
        props.output_text(fmt::format("Array size: {}", get_array_size()));
        for (uint32_t i = 0; i < get_array_size(); i++) {
            if (get_buffer(i) &&
                props.st_begin_child(std::to_string(i), fmt::format("Buffer {:04d}", i))) {
                get_buffer(i)->properties(props);
                props.st_end_child();
            }
        }
    }

    uint32_t get_array_size() const {
        return array_size;
    }

    merian::BufferHandle operator->() const {
        const merian::BufferHandle& buffer = get_buffer(0);
        assert(buffer);
        return buffer;
    }

    operator const merian::BufferHandle&() const {
        const merian::BufferHandle& buffer = get_buffer(0);
        assert(buffer);
        return buffer;
    }

    merian::Buffer& operator*() const {
        const merian::BufferHandle& buffer = get_buffer(0);
        assert(buffer);
        return *buffer;
    }

  private:
    const uint32_t array_size;
};

using BufferArrayResourceHandle = std::shared_ptr<BufferArrayResource>;

} // namespace merian
