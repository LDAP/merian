#pragma once

#include "merian-graph/graph/node.hpp"
#include "merian/vk/memory/staging_memory_manager.hpp"

#include <functional>
#include <mutex>

namespace merian {

// Outputs what a host uploaded last. The host methods are thread-safe.
class HostSource : public Node {
  public:
    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    NodeStatusFlags pre_process(const NodeIO& io, const NodeProcessInfo& info) override;

    NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

  protected:
    // Runs with mutex held.
    virtual void cmd_copy(const CommandBufferHandle& cmd,
                          const NodeIO& io,
                          const BufferHandle& staging_buffer,
                          const vk::DeviceSize staging_offset) = 0;

    // Runs create with mutex held; the created output receives the last upload.
    void create_output(const std::function<void()>& create);

    // update_layout runs with mutex held and returns whether the output must be recreated.
    void stage_upload(const vk::DeviceSize size,
                      const std::function<void(void* data)>& fill,
                      const std::function<bool()>& update_layout);

  private:
    std::mutex mutex;

    struct Staged {
        MemoryAllocationHandle memory;
        BufferHandle buffer;
        vk::DeviceSize offset = 0;
    };

    StagingMemoryManagerHandle staging;

    Staged current;
    bool pending = false;
    bool layout_changed = false;
};

} // namespace merian
