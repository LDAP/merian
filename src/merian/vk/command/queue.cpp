#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <mutex>
#include <optional>
#include <limits>

#include <vulkan/vulkan.hpp>

namespace merian {

Queue::Queue(const ContextHandle& context, uint32_t queue_family_index, uint32_t queue_index)
    : context(context), queue(context->device.getQueue(queue_family_index, queue_index)),
      queue_family_index(queue_family_index) {}

void Queue::submit(const vk::ArrayProxy<vk::SubmitInfo>& submit_infos, vk::Fence fence) {
    std::lock_guard<std::mutex> lock_guard(mutex);
    queue.submit(submit_infos, fence);
}

void Queue::submit(
    const vk::ArrayProxy<const vk::CommandBuffer>& cmds,
    const vk::Fence fence,
    const vk::ArrayProxy<vk::Semaphore>& signal_semaphores,
    const vk::ArrayProxy<vk::Semaphore>& wait_semaphores,
    const vk::ArrayProxy<vk::PipelineStageFlags>& wait_dst_stage_mask,
    const std::optional<VkTimelineSemaphoreSubmitInfo> timeline_semaphore_submit_info) {

    const vk::SubmitInfo submit_info{wait_semaphores, wait_dst_stage_mask, cmds, signal_semaphores,
                                     timeline_semaphore_submit_info.has_value()
                                         ? &timeline_semaphore_submit_info.value()
                                         : VK_NULL_HANDLE};
    submit(submit_info, fence);
}

void Queue::submit(
    const vk::ArrayProxy<const CommandBufferHandle>& cmds,
    const vk::Fence fence,
    const vk::ArrayProxy<vk::Semaphore>& signal_semaphores,
    const vk::ArrayProxy<vk::Semaphore>& wait_semaphores,
    const vk::ArrayProxy<vk::PipelineStageFlags>& wait_dst_stage_mask,
    const std::optional<VkTimelineSemaphoreSubmitInfo> timeline_semaphore_submit_info) {

    std::vector<vk::CommandBuffer> vk_cmds(cmds.size());
    std::transform(cmds.begin(), cmds.end(), vk_cmds.begin(),
                   [&](auto& c) { return c->get_command_buffer(); });

    const vk::SubmitInfo submit_info{
        wait_semaphores, wait_dst_stage_mask, vk_cmds, signal_semaphores,
        timeline_semaphore_submit_info.has_value() ? &timeline_semaphore_submit_info.value()
                                                   : VK_NULL_HANDLE};
    submit(submit_info, fence);
}

void Queue::submit_wait(const vk::ArrayProxy<vk::SubmitInfo>& submit_infos, const vk::Fence fence) {
    submit(submit_infos, fence);

    if (fence) {
        check_result(
            context->device.waitForFences(fence, VK_TRUE, std::numeric_limits<uint64_t>::max()),
            "failed waiting for fence");
    } else {
        wait_idle();
    }
}

void Queue::submit_wait(
    const vk::ArrayProxy<const vk::CommandBuffer>& cmds,
    const vk::Fence fence,
    const vk::ArrayProxy<vk::Semaphore>& signal_semaphores,
    const vk::ArrayProxy<vk::Semaphore>& wait_semaphores,
    const vk::ArrayProxy<vk::PipelineStageFlags>& wait_dst_stage_mask,
    const std::optional<VkTimelineSemaphoreSubmitInfo> timeline_semaphore_submit_info) {

    const vk::SubmitInfo submit_info{wait_semaphores, wait_dst_stage_mask, cmds, signal_semaphores,
                                     timeline_semaphore_submit_info.has_value()
                                         ? &timeline_semaphore_submit_info.value()
                                         : VK_NULL_HANDLE};
    submit_wait(submit_info, fence);
}

void Queue::submit_wait(
    const vk::ArrayProxy<const CommandBufferHandle>& cmds,
    const vk::Fence fence,
    const vk::ArrayProxy<vk::Semaphore>& signal_semaphores,
    const vk::ArrayProxy<vk::Semaphore>& wait_semaphores,
    const vk::ArrayProxy<vk::PipelineStageFlags>& wait_dst_stage_mask,
    const std::optional<VkTimelineSemaphoreSubmitInfo> timeline_semaphore_submit_info) {

    std::vector<vk::CommandBuffer> vk_cmds(cmds.size());
    std::transform(cmds.begin(), cmds.end(), vk_cmds.begin(),
                   [&](auto& c) { return c->get_command_buffer(); });

    const vk::SubmitInfo submit_info{
        wait_semaphores, wait_dst_stage_mask, vk_cmds, signal_semaphores,
        timeline_semaphore_submit_info.has_value() ? &timeline_semaphore_submit_info.value()
                                                   : VK_NULL_HANDLE};
    submit_wait(submit_info, fence);
}

void Queue::submit_wait(const CommandPoolHandle& cmd_pool,
                        const std::function<void(const CommandBufferHandle& cmd)>& cmd_function) {
    const CommandBufferHandle cmd = std::make_shared<CommandBuffer>(cmd_pool);
    cmd->begin();
    cmd_function(cmd);
    cmd->end();
    const vk::Fence fence = context->device.createFence({});
    submit_wait(cmd, fence);
    context->device.destroyFence(fence);
}

void Queue::submit_wait(const std::function<void(const CommandBufferHandle& cmd)>& cmd_function) {
    const CommandPoolHandle cmd_pool = std::make_shared<CommandPool>(shared_from_this());
    submit_wait(cmd_pool, cmd_function);
}

vk::Result Queue::present(const vk::PresentInfoKHR& present_info) {
    std::lock_guard<std::mutex> lock_guard(mutex);
    return queue.presentKHR(present_info);
}

vk::Result Queue::present(const vk::PresentInfoKHR&& present_info) {
    std::lock_guard<std::mutex> lock_guard(mutex);
    return queue.presentKHR(present_info);
}

void Queue::wait_idle() {
    std::lock_guard<std::mutex> lock_guard(mutex);
    queue.waitIdle();
}

} // namespace merian
