#pragma once

#include "merian/utils/function.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/sync/ring_fences.hpp"

#include "graph_run.hpp"
#include "node.hpp"

#include <cstdint>

namespace merian {

/**
 * @brief      A Vulkan processing graph.
 *
 * @tparam     RING_SIZE  Controls the amount of in-flight processing (frames-in-flight).
 */
template <uint32_t RING_SIZE = 2> class Graph {
  private:
    struct FrameData {
        // The command pool for the current frame.
        // We do not use RingCommandPool here since we might want to add a more custom
        // setup later (multi-threaded, multi-queues,...)
        std::shared_ptr<CommandPool> command_pool;
        merian::StagingMemoryManager::SetID staging_set_id{};
        GraphRun graph_run;
    };

  public:
    Graph(const SharedContext& context, const ResourceAllocatorHandle& resource_allocator)
        : context(context), resource_allocator(resource_allocator), queue(context->get_queue_GCT()),
          ring_fences(context) {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            ring_fences.get_ring_data(i).command_pool = std::make_shared<CommandPool>(queue);
        }
    }

    // Runs one iteration of the graph.
    // If necessary, the graph is automatically built.
    // The execution is blocked until the fence according to this frame is signaled.
    // Interaction with the run is possible using the callbacks.
    void run() {
        // PREPARE RUN: wait for fence, release resources, reset cmd pool

        // wait for the in-flight processing to finish
        auto& frame_data = ring_fences.next_cycle_wait_and_get();
        // now we can release the resources from staging space and reset the command pool
        resource_allocator->getStaging()->releaseResourceSet(frame_data.user_data.staging_set_id);
        const std::shared_ptr<CommandPool>& cmd_pool = frame_data.user_data.command_pool;
        cmd_pool->reset();
        GraphRun& run = frame_data.user_data.graph_run;
        run.reset(nullptr);
        run_if_set<void, GraphRun&>(on_run_starting, run);
        const vk::CommandBuffer cmd = cmd_pool->create_and_begin();

        // EXECUTE RUN

        // FINISH RUN: submit

        run_if_set<void, GraphRun&>(on_pre_submit, run);
        cmd_pool->end_all();
        frame_data.user_data.staging_set_id =
            resource_allocator->getStaging()->finalizeResourceSet();
        queue->submit(cmd_pool, frame_data.fence, run.get_signal_semaphores(),
                      run.get_wait_semaphores(), run.get_wait_stages(),
                      run.get_timeline_semaphore_submit_info());
        run.execute_callbacks(queue);
        run_if_set(on_post_submit);
    }

  private:
    // General stuff
    const SharedContext context;
    const ResourceAllocatorHandle resource_allocator;
    const QueueHandle queue;

    // Outside callbacks
    std::function<void(GraphRun& graph_run)> on_run_starting;
    std::function<void(GraphRun& graph_run)> on_pre_submit;
    std::function<void()> on_post_submit;

    // Per-frame data management
    merian::RingFences<RING_SIZE, FrameData> ring_fences;
};

} // namespace merian
