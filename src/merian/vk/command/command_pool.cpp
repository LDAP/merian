#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/command/queue.hpp"
#include <spdlog/spdlog.h>

namespace merian {

CommandPool::CommandPool(const std::shared_ptr<Queue> queue,
                         vk::CommandPoolCreateFlags create_flags,
                         const uint32_t cache_size)
    : CommandPool(queue->get_context(), queue->get_queue_family_index(), create_flags, cache_size) {
}

CommandPool::CommandPool(const SharedContext context,
                         uint32_t queue_family_index,
                         vk::CommandPoolCreateFlags create_flags,
                         const uint32_t cache_size)
    : context(context), cache_size(cache_size) {
    vk::CommandPoolCreateInfo info{create_flags, queue_family_index};
    pool = context->device.createCommandPool(info);
    SPDLOG_DEBUG("create command pool ({})", fmt::ptr(this));
};

CommandPool::~CommandPool() {
    context->device.waitIdle();
    SPDLOG_DEBUG("destroy command pool ({})", fmt::ptr(this));
    if (!cache_primary_cmds.empty())
        context->device.freeCommandBuffers(pool, cache_primary_cmds);
    reset();
    context->device.destroyCommandPool(pool);
};

vk::CommandBuffer CommandPool::create(vk::CommandBufferLevel level,
                                      bool begin,
                                      vk::CommandBufferUsageFlags flags,
                                      const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    vk::CommandBuffer cmd;
    if (level == vk::CommandBufferLevel::ePrimary) {
        if (!cache_primary_cmds.empty()) {
            inuse_primary_cmds.push_back(cache_primary_cmds.back());
            cache_primary_cmds.pop_back();
        } else {
            inuse_primary_cmds.emplace_back();
            vk::CommandBufferAllocateInfo info{pool, level, 1};
            check_result(context->device.allocateCommandBuffers(&info, &inuse_primary_cmds.back()),
                         "could not allocate command buffer");
        }
        cmd = inuse_primary_cmds.back();
    } else {
        // always allocate
        vk::CommandBufferAllocateInfo info{pool, level, 1};
        check_result(context->device.allocateCommandBuffers(&info, &cmd),
                         "could not allocate command buffer");
    }


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

    if (level == vk::CommandBufferLevel::ePrimary)
        insert_all(inuse_primary_cmds, allocated);

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
    // keep some cmds on hand
    while (cache_primary_cmds.size() < cache_size && !inuse_primary_cmds.empty()) {
        cache_primary_cmds.push_back(std::move(inuse_primary_cmds.back()));
        inuse_primary_cmds.pop_back();
    }

    if (!inuse_primary_cmds.empty()) {
        // free the rest
        context->device.freeCommandBuffers(pool, inuse_primary_cmds);
        inuse_primary_cmds.clear();
    }

    context->device.resetCommandPool(pool);
}

const std::vector<vk::CommandBuffer>& CommandPool::get_command_buffers() const {
    return inuse_primary_cmds;
}

void CommandPool::end_all() {
    for (vk::CommandBuffer& cmd : inuse_primary_cmds) {
        cmd.end();
    }
}

} // namespace merian
