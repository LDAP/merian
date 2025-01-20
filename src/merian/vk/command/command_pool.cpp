#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/command/queue.hpp"

#include <spdlog/spdlog.h>

namespace merian {

CommandPool::CommandPool(const ContextHandle& context)
    : context(context), queue_family_index(-1u) {}

CommandPool::CommandPool(const QueueHandle& queue, vk::CommandPoolCreateFlags create_flags)
    : CommandPool(queue->get_context(), queue->get_queue_family_index(), create_flags) {}

CommandPool::CommandPool(const ContextHandle& context,
                         const uint32_t queue_family_index,
                         const vk::CommandPoolCreateFlags create_flags)
    : context(context), queue_family_index(queue_family_index) {
    vk::CommandPoolCreateInfo info{create_flags, queue_family_index};
    pool = context->device.createCommandPool(info);
    SPDLOG_DEBUG("create command pool ({})", fmt::ptr(static_cast<VkCommandPool>(pool)));
};

CommandPool::~CommandPool() {
    if (pool) {
        // special case for Cachign Command Pool
        SPDLOG_DEBUG("destroy command pool ({})", fmt::ptr(static_cast<VkCommandPool>(pool)));
        reset();
        context->device.destroyCommandPool(pool);
    }
};

uint32_t CommandPool::get_queue_family_index() const noexcept {
    return queue_family_index;
}

const vk::CommandPool& CommandPool::get_pool() const noexcept {
    return pool;
}

CommandPool::operator const vk::CommandPool&() const noexcept {
    return pool;
}

const vk::CommandPool& CommandPool::operator*() const noexcept {
    return pool;
}

// Frees command buffers, resets command pool
void CommandPool::reset() {
    context->device.resetCommandPool(pool);
    objects_in_use.clear();
}

} // namespace merian
