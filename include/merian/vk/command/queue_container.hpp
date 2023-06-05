#pragma once

#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <mutex>
#include <vulkan/vulkan.hpp>

namespace merian {

/* A container that holds a queue together with a mutex and provides utility functions.
 *
 * All submits are protected using a mutex. When using *_wait the queue is blocked until the queue
 * is idle.
 */
class QueueContainer : public std::enable_shared_from_this<QueueContainer> {
  public:
    QueueContainer() = delete;

    QueueContainer(const SharedContext& context, uint32_t queue_family_index, uint32_t queue_index);

    void submit(const std::shared_ptr<CommandPool>& pool,
                const vk::Fence fence,
                const std::vector<vk::Semaphore>& wait_semaphores = {},
                const std::vector<vk::Semaphore>& signal_semaphores = {},
                const vk::PipelineStageFlags& wait_dst_stage_mask = {});

    void submit(const std::vector<vk::CommandBuffer>& command_buffers,
                vk::Fence fence = VK_NULL_HANDLE,
                const std::vector<vk::Semaphore>& wait_semaphores = {},
                const std::vector<vk::Semaphore>& signal_semaphores = {},
                const vk::PipelineStageFlags& wait_dst_stage_mask = {});

    void submit(const vk::CommandBuffer& command_buffer, vk::Fence fence = VK_NULL_HANDLE);

    void submit(const vk::SubmitInfo& submit_info,
                vk::Fence fence = VK_NULL_HANDLE,
                uint32_t submit_count = 1);

    void submit(const std::vector<vk::SubmitInfo>& submit_infos, vk::Fence fence = VK_NULL_HANDLE);

    void submit_wait(const std::shared_ptr<CommandPool>& pool,
                     const vk::Fence fence = {},
                     const std::vector<vk::Semaphore>& wait_semaphores = {},
                     const std::vector<vk::PipelineStageFlags>& wait_dst_stage_mask = {},
                     const std::vector<vk::Semaphore>& signal_semaphores = {});

    // Submits the command buffers then waits using waitIdle(), try to not use the _wait variants
    void submit_wait(const std::vector<vk::CommandBuffer>& command_buffers,
                     vk::Fence fence = VK_NULL_HANDLE,
                     const std::vector<vk::Semaphore>& wait_semaphores = {},
                     const std::vector<vk::Semaphore>& signal_semaphores = {},
                     const vk::PipelineStageFlags& wait_dst_stage_mask = {});

    // Submits the command buffers then waits using waitIdle(), try to not use the _wait variants
    void submit_wait(const vk::CommandBuffer& command_buffer, vk::Fence fence = VK_NULL_HANDLE);

    // Submits then waits using waitIdle(), try to not use the _wait variants
    void submit_wait(const vk::SubmitInfo& submit_info, vk::Fence fence = VK_NULL_HANDLE);

    void present(const vk::PresentInfoKHR& present_info);

    const SharedContext& get_context() const {
        return context;
    }

    uint32_t get_queue_family_index() const {
        return queue_family_index;
    }

    operator uint32_t() const {
        return queue_family_index;
    }

    // Returns the queue. Try to not use the queue directly.
    vk::Queue get_queue() const {
        return queue;
    }

    // Returns the queue. Try to not use the queue directly
    operator vk::Queue() const {
        return queue;
    }

  private:
    const SharedContext context;
    // Try to not use the queue directly
    const vk::Queue queue;
    const uint32_t queue_family_index;
    std::mutex mutex;
};

} // namespace merian
