#pragma once

#include "merian/vk/command/command_pool.hpp"

#include <vulkan/vulkan.hpp>

#include <mutex>

namespace merian {

class Queue;
using QueueHandle = std::shared_ptr<Queue>;
class CommandPool;
using CommandPoolHandle = std::shared_ptr<CommandPool>;

/* A container that holds a queue together with a mutex and provides utility functions.
 *
 * All submits are protected using a mutex. When using *_wait the queue is blocked until the queue
 * is idle.
 */
class Queue : public std::enable_shared_from_this<Queue> {
  public:
    Queue() = delete;

    Queue(const ContextHandle& context, uint32_t queue_family_index, uint32_t queue_index);

    // ------------------------------------------------------------------

    void submit(const vk::ArrayProxy<vk::SubmitInfo>& submit_infos,
                vk::Fence fence = VK_NULL_HANDLE);

    void submit(const vk::ArrayProxy<const vk::CommandBuffer>& cmds,
                const vk::Fence fence = VK_NULL_HANDLE,
                const vk::ArrayProxy<vk::Semaphore>& signal_semaphores = {},
                const vk::ArrayProxy<vk::Semaphore>& wait_semaphores = {},
                const vk::ArrayProxy<vk::PipelineStageFlags>& wait_dst_stage_mask = {},
                const std::optional<VkTimelineSemaphoreSubmitInfo> timeline_semaphore_submit_info =
                    std::nullopt);

    void submit(const vk::ArrayProxy<const CommandBufferHandle>& cmds,
                const vk::Fence fence = VK_NULL_HANDLE,
                const vk::ArrayProxy<vk::Semaphore>& signal_semaphores = {},
                const vk::ArrayProxy<vk::Semaphore>& wait_semaphores = {},
                const vk::ArrayProxy<vk::PipelineStageFlags>& wait_dst_stage_mask = {},
                const std::optional<VkTimelineSemaphoreSubmitInfo> timeline_semaphore_submit_info =
                    std::nullopt);

    // ------------------------------------------------------------------

    // Submits, then waits using the fence or wait_idle(). Try to not use the _wait variants.
    void submit_wait(const vk::ArrayProxy<vk::SubmitInfo>& submit_infos,
                     const vk::Fence fence = VK_NULL_HANDLE);

    // Submits all command buffers of the pool, then waits using the fence or wait_idle().
    void submit_wait(const vk::ArrayProxy<const vk::CommandBuffer>& cmds,
                     const vk::Fence fence = VK_NULL_HANDLE,
                     const vk::ArrayProxy<vk::Semaphore>& signal_semaphores = {},
                     const vk::ArrayProxy<vk::Semaphore>& wait_semaphores = {},
                     const vk::ArrayProxy<vk::PipelineStageFlags>& wait_dst_stage_mask = {},
                     const std::optional<VkTimelineSemaphoreSubmitInfo>
                         timeline_semaphore_submit_info = std::nullopt);

    // Submits all command buffers of the pool, then waits using the fence or wait_idle().
    void submit_wait(const vk::ArrayProxy<const CommandBufferHandle>& cmds,
                     const vk::Fence fence = VK_NULL_HANDLE,
                     const vk::ArrayProxy<vk::Semaphore>& signal_semaphores = {},
                     const vk::ArrayProxy<vk::Semaphore>& wait_semaphores = {},
                     const vk::ArrayProxy<vk::PipelineStageFlags>& wait_dst_stage_mask = {},
                     const std::optional<VkTimelineSemaphoreSubmitInfo>
                         timeline_semaphore_submit_info = std::nullopt);

    // ------------------------------------------------------------------

    // Utility function that
    // - Creates and begins a command buffer
    // - Records commands using the supplied cmd_function
    // - Submits the command buffer
    // - Waits for the execution to finish
    void submit_wait(const CommandPoolHandle& cmd_pool,
                     const std::function<void(const CommandBufferHandle& cmd)>& cmd_function);

    // Utility function that
    // - Creates and begins a command pool and command buffer
    // - Records commands using the supplied cmd_function
    // - Submits the command buffer
    // - Waits for the execution to finish
    void submit_wait(const std::function<void(const CommandBufferHandle& cmd)>& cmd_function);

    // ------------------------------------------------------------------

    vk::Result present(const vk::PresentInfoKHR& present_info);

    vk::Result present(const vk::PresentInfoKHR&& present_info);

    void wait_idle();

    const ContextHandle& get_context() const {
        return context;
    }

    uint32_t get_queue_family_index() const {
        return queue_family_index;
    }

    vk::QueueFamilyProperties get_queue_family_properties() const {
        return context->physical_device.physical_device
            .getQueueFamilyProperties()[queue_family_index];
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
    const ContextHandle context;
    // Try to not use the queue directly
    const vk::Queue queue;
    const uint32_t queue_family_index;
    std::mutex mutex;
};

} // namespace merian
