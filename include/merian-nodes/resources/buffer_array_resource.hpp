#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class BufferArrayResource : public GraphResource {
    friend class VkBufferArrayOut;
    friend class VkBufferArrayIn;
    friend class ManagedVkBufferOut;
    friend class ManagedVkBufferIn;

  public:
    BufferArrayResource(std::vector<merian::BufferHandle>& buffers,
                        const merian::BufferHandle& dummy_buffer,
                        const vk::PipelineStageFlags2 input_stage_flags,
                        const vk::AccessFlags2 input_access_flags)
        : input_stage_flags(input_stage_flags), input_access_flags(input_access_flags),
          buffers(buffers), dummy_buffer(dummy_buffer) {

        for (uint32_t i = 0; i < buffers.size(); i++) {
            current_updates.push_back(i);
        }
    }

    // Automatically inserts barrier, can be nullptr to reset to dummy buffer.
    void set(const uint32_t index,
             const merian::BufferHandle& buffer,
             const merian::CommandBufferHandle& cmd,
             const vk::AccessFlags2 prior_access_flags,
             const vk::PipelineStageFlags2 prior_pipeline_stages) {
        assert(index < buffers.size());

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
    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;

  private:
    // the updates to "buffers" are recorded here.
    std::vector<uint32_t> current_updates;
    // then flushed to here to wait for the graph to apply descriptor updates.
    std::vector<uint32_t> pending_updates;

    std::vector<merian::BufferHandle>& buffers;

    const merian::BufferHandle dummy_buffer;
};

} // namespace merian_nodes
