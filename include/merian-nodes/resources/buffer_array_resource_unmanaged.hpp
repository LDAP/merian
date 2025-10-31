#pragma once

#include "merian-nodes/resources/buffer_array_resource.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/memory/resource_allocations.hpp"

namespace merian {

class UnmanagedBufferArrayResource : public BufferArrayResource {
    friend class VkBufferIn;
    friend class ManagedVkBufferOut;
    friend class UnmanagedVkBufferOut;

  public:
    UnmanagedBufferArrayResource(const uint32_t array_size,
                                 const vk::BufferUsageFlags buffer_usage_flags,
                                 const vk::PipelineStageFlags2 input_stage_flags,
                                 const vk::AccessFlags2 input_access_flags,
                                 std::vector<merian::BufferHandle>& buffers)
        : BufferArrayResource(
              array_size, buffer_usage_flags, input_stage_flags, input_access_flags),
          buffers(buffers) {}

    void set(const uint32_t index,
             const merian::BufferHandle& buffer,
             const merian::CommandBufferHandle& cmd,
             const vk::AccessFlags2 prior_access_flags,
             const vk::PipelineStageFlags2 prior_pipeline_stages) {
        assert(index < buffers.size());
        assert((buffer->get_usage_flags() & get_buffer_usage_flags()) == get_buffer_usage_flags());

        if (buffers[index] != buffer) {
            buffers[index] = buffer;
            queue_descriptor_update(index);
        }

        if (buffer && prior_access_flags) {
            const vk::BufferMemoryBarrier2 buf_bar =
                buffer->buffer_barrier2(prior_pipeline_stages, get_input_stage_flags(),
                                        prior_access_flags, get_input_access_flags());

            cmd->barrier(buf_bar);
        }
    }

    // can be nullptr
    const merian::BufferHandle& get_buffer(const uint32_t index) const override {
        assert(index < buffers.size());
        return buffers[index];
    }

  private:
    std::vector<merian::BufferHandle>& buffers;
};

} // namespace merian
