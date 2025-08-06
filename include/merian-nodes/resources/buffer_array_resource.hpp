#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/command/command_buffer.hpp"
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
        : buffer_usage_flags(buffer_usage_flags), input_stage_flags(input_stage_flags),
          input_access_flags(input_access_flags), buffers(array_size) {

        for (uint32_t i = 0; i < buffers.size(); i++) {
            current_updates.push_back(i);
        }
    }

    void set(const uint32_t index,
             const merian::BufferHandle& buffer,
             const merian::CommandBufferHandle& cmd,
             const vk::AccessFlags2 prior_access_flags,
             const vk::PipelineStageFlags2 prior_pipeline_stages) {
        assert(index < buffers.size());
        assert((buffer->get_usage_flags() & buffer_usage_flags) == buffer_usage_flags);

        if (buffers[index] != buffer) {
            buffers[index] = buffer;
            current_updates.push_back(index);
        }

        if (buffer && prior_access_flags) {
            const vk::BufferMemoryBarrier2 buf_bar = buffer->buffer_barrier2(
                prior_pipeline_stages, input_stage_flags, prior_access_flags, input_access_flags);

            cmd->barrier(buf_bar);
        }
    }

    const merian::BufferHandle& get(const uint32_t index) const {
        assert(index < buffers.size());

        return buffers[index];
    }

    void properties(merian::Properties& props) override {
        props.output_text(
            fmt::format("Array size: {}\nCurrent updates: {}\nPending updates: {}\nInput access "
                        "flags: {}\nInput pipeline stages: {}",
                        buffers.size(), current_updates.size(), pending_updates.size(),
                        vk::to_string(input_access_flags), vk::to_string(input_stage_flags)));
        for (uint32_t i = 0; i < buffers.size(); i++) {
            if (buffers[i] &&
                props.st_begin_child(std::to_string(i), fmt::format("Buffer {:04d}", i))) {
                buffers[i]->properties(props);
                props.st_end_child();
            }
        }
    }

    uint32_t get_array_size() const {
        return buffers.size();
    }

    merian::BufferHandle operator->() const {
        assert(!buffers.empty());
        return buffers[0];
    }

    operator merian::BufferHandle() const {
        assert(!buffers.empty());
        return buffers[0];
    }

    merian::Buffer& operator*() const {
        assert(!buffers.empty());
        return *buffers[0];
    }

  public:
    const vk::BufferUsageFlags buffer_usage_flags;
    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;

  private:
    // the updates to "buffers" are recorded here.
    std::vector<uint32_t> current_updates;
    // then flushed to here to wait for the graph to apply descriptor updates.
    std::vector<uint32_t> pending_updates;

    std::vector<merian::BufferHandle> buffers;
};

} // namespace merian_nodes
