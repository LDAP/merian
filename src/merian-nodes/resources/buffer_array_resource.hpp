#pragma once

#include "merian-nodes/graph/resource.hpp"

#include "merian/vk/memory/resource_allocations.hpp"

namespace merian_nodes {

class BufferArrayResource : public GraphResource {
    friend class BufferArrayOut;
    friend class BufferArrayIn;

  public:
    BufferArrayResource(std::vector<merian::BufferHandle>& buffers,
                        const uint32_t ring_size,
                        const merian::BufferHandle& dummy_buffer,
                        const vk::PipelineStageFlags2 input_stage_flags,
                        const vk::AccessFlags2 input_access_flags)
        : buffers(buffers), dummy_buffer(dummy_buffer), input_stage_flags(input_stage_flags),
          input_access_flags(input_access_flags) {
        in_flight_buffers.assign(ring_size, std::vector<merian::BufferHandle>(buffers.size()));

        for (uint32_t i = 0; i < buffers.size(); i++) {
            current_updates.push_back(i);
        }
    }

    // Automatically inserts barrier, can be nullptr to reset to dummy buffer.
    void set(const uint32_t index,
             const merian::BufferHandle& buffer,
             const vk::CommandBuffer cmd,
             const vk::AccessFlags2 prior_access_flags,
             const vk::PipelineStageFlags2 prior_pipeline_stages) {
        buffers[index] = buffer;
        current_updates.push_back(index);

        if (buffer) {
            const vk::BufferMemoryBarrier2 buf_bar = buffer->buffer_barrier2(
                prior_pipeline_stages, input_stage_flags, prior_access_flags, input_access_flags);

            cmd.pipelineBarrier2(vk::DependencyInfo{{}, {}, buf_bar, {}});
        }
    }

    const merian::BufferHandle& get(const uint32_t index) const {
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

  private:
    // the updates to "buffers" are recorded here.
    std::vector<uint32_t> current_updates;
    // then flushed to here to wait for the graph to apply descriptor updates.
    std::vector<uint32_t> pending_updates;

    std::vector<merian::BufferHandle>& buffers;
    // on_post_process copy here to keep alive
    std::vector<std::vector<merian::BufferHandle>> in_flight_buffers;

    const merian::BufferHandle dummy_buffer;

    // combined pipeline stage flags of all inputs
    const vk::PipelineStageFlags2 input_stage_flags;
    // combined access flags of all inputs
    const vk::AccessFlags2 input_access_flags;
};

} // namespace merian_nodes
