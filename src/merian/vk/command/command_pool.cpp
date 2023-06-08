#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/command/queue_container.hpp"
#include <spdlog/spdlog.h>

namespace merian {

CommandPool::CommandPool(const std::shared_ptr<QueueContainer> queue,
                         vk::CommandPoolCreateFlags create_flags)
    : CommandPool(queue->get_context(), queue->get_queue_family_index(), create_flags) {}

CommandPool::CommandPool(const SharedContext context,
                         uint32_t queue_family_index,
                         vk::CommandPoolCreateFlags create_flags)
    : context(context) {
    vk::CommandPoolCreateInfo info{create_flags, queue_family_index};
    pool = context->device.createCommandPool(info);
    SPDLOG_DEBUG("create command pool ({})", fmt::ptr(this));
};

CommandPool::~CommandPool() {
    context->device.waitIdle();
    SPDLOG_DEBUG("destroy command pool ({})", fmt::ptr(this));
    reset();
    context->device.destroyCommandPool(pool);
};

vk::CommandBuffer CommandPool::create(vk::CommandBufferLevel level,
                                      bool begin,
                                      vk::CommandBufferUsageFlags flags,
                                      const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    cmds.emplace_back();
    vk::CommandBuffer& cmd = cmds[cmds.size() - 1];

    vk::CommandBufferAllocateInfo info{pool, level, 1};
    check_result(context->device.allocateCommandBuffers(&info, &cmd),
                 "could not allocate command buffer");

    if (begin) {
        vk::CommandBufferBeginInfo info{flags, pInheritanceInfo};
        cmd.begin(info);
    }

    return cmd;
}

std::vector<vk::CommandBuffer>
CommandPool::create_multiple(vk::CommandBufferLevel level,
                             uint32_t count,
                             bool begin,
                             vk::CommandBufferUsageFlags flags,
                             const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    vk::CommandBufferAllocateInfo info{pool, level, count};

    std::vector<vk::CommandBuffer> allocated = context->device.allocateCommandBuffers(info);
    insert_all(cmds, allocated);

    if (begin) {
        vk::CommandBufferBeginInfo info{flags, pInheritanceInfo};
        for (vk::CommandBuffer& cmd : allocated) {
            cmd.begin(info);
        }
    }

    return allocated;
}

vk::CommandPool& CommandPool::get_pool() {
    return pool;
}

// Frees command buffers, resets command pool
void CommandPool::reset() {
    if (has_command_buffers()) {
        context->device.freeCommandBuffers(pool, cmds);
        cmds.clear();
    }
    context->device.resetCommandPool(pool);
}

bool CommandPool::has_command_buffers() {
    return !cmds.empty();
}

// Ends all command buffers
void CommandPool::end_all() {
    for (vk::CommandBuffer& cmd : cmds) {
        cmd.end();
    }
}

const std::vector<vk::CommandBuffer>& CommandPool::get_command_buffers() const {
    return cmds;
}

} // namespace merian
