#pragma once

#include "merian-nodes/graph/resource.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class BufferArrayResource : public GraphResource {
    friend class VkBufferIn;
    friend class ManagedVkBufferOut;
    friend class UnmanagedVkBufferOut;

  public:
    BufferArrayResource(const uint32_t array_size,
                        const vk::BufferUsageFlags buffer_usage_flags,
                        const vk::PipelineStageFlags2 input_stage_flags,
                        const vk::AccessFlags2 input_access_flags)
        : array_size(array_size), buffer_usage_flags(buffer_usage_flags),
          input_stage_flags(input_stage_flags), input_access_flags(input_access_flags) {

        for (uint32_t i = 0; i < array_size; i++) {
            current_updates.push_back(i);
        }
    }

    // can be nullptr
    virtual const merian::BufferHandle& get_buffer(const uint32_t index) const = 0;

    void properties(merian::Properties& props) override {
        props.output_text(
            fmt::format("Array size: {}\nCurrent updates: {}\nPending updates: {}\nInput access "
                        "flags: {}\nInput pipeline stages: {}",
                        get_array_size(), current_updates.size(), pending_updates.size(),
                        vk::to_string(input_access_flags), vk::to_string(input_stage_flags)));
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

    // -------------------------------------------------

    vk::BufferUsageFlags get_buffer_usage_flags() const {
        return buffer_usage_flags;
    }

    vk::PipelineStageFlags2 get_input_stage_flags() const {
        return input_stage_flags;
    }

    vk::AccessFlags2 get_input_access_flags() const {
        return input_access_flags;
    }

  protected:
    void queue_descriptor_update(const uint32_t array_index) {
        current_updates.emplace_back(array_index);
    }

  private:
    const uint32_t array_size;

    const vk::BufferUsageFlags buffer_usage_flags;
    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;

    // the updates to "buffers" are recorded here.
    std::vector<uint32_t> current_updates;
    // then flushed to here to wait for the graph to apply descriptor updates.
    std::vector<uint32_t> pending_updates;
};

} // namespace merian_nodes
