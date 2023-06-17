#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <mutex>
#include <vulkan/vulkan.hpp>

namespace merian {

Queue::Queue(const SharedContext& context,
                               uint32_t queue_family_index,
                               uint32_t queue_index)
    : context(context), queue(context->device.getQueue(queue_family_index, queue_index)),
      queue_family_index(queue_family_index) {}

void Queue::submit(const std::shared_ptr<CommandPool>& pool,
                            const vk::Fence fence,
                            const std::vector<vk::Semaphore>& signal_semaphores,
                            const std::vector<vk::Semaphore>& wait_semaphores,
                            const std::vector<vk::PipelineStageFlags>& wait_dst_stage_mask) {
    vk::SubmitInfo submit_info{wait_semaphores, wait_dst_stage_mask, pool->get_command_buffers(),
                               signal_semaphores};
    submit(submit_info, fence);
}

void Queue::submit(const std::vector<vk::CommandBuffer>& command_buffers,
                            vk::Fence fence,
                            const std::vector<vk::Semaphore>& signal_semaphores,
                            const std::vector<vk::Semaphore>& wait_semaphores,
                            const std::vector<vk::PipelineStageFlags>& wait_dst_stage_mask) {
    vk::SubmitInfo submit_info{wait_semaphores, wait_dst_stage_mask, command_buffers,
                               signal_semaphores};
    submit(submit_info, fence);
}

void Queue::submit(const vk::CommandBuffer& command_buffer, vk::Fence fence) {
    vk::SubmitInfo submit_info{
        {}, {}, {}, 1, &command_buffer,
    };
    submit(submit_info, fence);
}

void Queue::submit(const vk::SubmitInfo& submit_info,
                            vk::Fence fence,
                            uint32_t submit_count) {
    std::lock_guard<std::mutex> lock_guard(mutex);
    check_result(queue.submit(submit_count, &submit_info, fence), "queue submit failed");
}

void Queue::submit(const std::vector<vk::SubmitInfo>& submit_infos, vk::Fence fence) {
    std::lock_guard<std::mutex> lock_guard(mutex);
    queue.submit(submit_infos, fence);
}

void Queue::submit_wait(const std::shared_ptr<CommandPool>& pool,
                                 const vk::Fence fence,
                                 const std::vector<vk::Semaphore>& signal_semaphores,
                                 const std::vector<vk::Semaphore>& wait_semaphores,
                                 const std::vector<vk::PipelineStageFlags>& wait_dst_stage_mask) {
    vk::SubmitInfo submit_info{wait_semaphores, wait_dst_stage_mask, pool->get_command_buffers(),
                               signal_semaphores};
    submit_wait(submit_info, fence);
}

// Submits the command buffers then waits using waitIdle(), try to not use the _wait variants
void Queue::submit_wait(const std::vector<vk::CommandBuffer>& command_buffers,
                                 vk::Fence fence,
                                 const std::vector<vk::Semaphore>& signal_semaphores,
                                 const std::vector<vk::Semaphore>& wait_semaphores,
                                 const vk::PipelineStageFlags& wait_dst_stage_mask) {
    vk::SubmitInfo submit_info{wait_semaphores, wait_dst_stage_mask, command_buffers,
                               signal_semaphores};
    submit(submit_info, fence);
}

// Submits the command buffers then waits using waitIdle(), try to not use the _wait variants
void Queue::submit_wait(const vk::CommandBuffer& command_buffer, vk::Fence fence) {
    vk::SubmitInfo submit_info{
        {}, {}, {}, 1, &command_buffer,
    };
    submit_wait(submit_info, fence);
}

// Submits then waits using waitIdle(), try to not use the _wait variants
void Queue::submit_wait(const vk::SubmitInfo& submit_info, vk::Fence fence) {
    std::lock_guard<std::mutex> lock_guard(mutex);
    check_result(queue.submit(1, &submit_info, fence), "queue submit failed");
    queue.waitIdle();
}

void Queue::present(const vk::PresentInfoKHR& present_info) {
    std::lock_guard<std::mutex> lock_guard(mutex);
    check_result(queue.presentKHR(&present_info), "present failed");
}

void Queue::wait_idle() {
    std::lock_guard<std::mutex> lock_guard(mutex);
    queue.waitIdle();
}

} // namespace merian
