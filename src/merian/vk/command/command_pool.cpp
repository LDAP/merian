#include "merian/vk/command/command_pool.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/command/queue.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <spdlog/spdlog.h>

namespace merian {

CommandPool::CommandPool(const std::shared_ptr<Queue> queue,
                         vk::CommandPoolCreateFlags create_flags)
    : CommandPool(queue->get_context(), queue->get_queue_family_index(), create_flags) {}

CommandPool::CommandPool(const ContextHandle context,
                         uint32_t queue_family_index,
                         vk::CommandPoolCreateFlags create_flags)
    : context(context), queue_family_index(queue_family_index) {
    vk::CommandPoolCreateInfo info{create_flags, queue_family_index};
    pool = context->device.createCommandPool(info);
    SPDLOG_DEBUG("create command pool ({})", fmt::ptr(this));
};

CommandPool::~CommandPool() {
    SPDLOG_DEBUG("destroy command pool ({})", fmt::ptr(this));
    reset();

    for (auto& cached_cmds : {cache_primary_cmds, cache_secondary_cmds}) {
        if (!cached_cmds.empty())
            context->device.freeCommandBuffers(pool, cached_cmds);
    }

    context->device.destroyCommandPool(pool);
};

vk::CommandBuffer CommandPool::create(const vk::CommandBufferLevel level,
                                      const bool begin,
                                      const vk::CommandBufferUsageFlags flags,
                                      const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    std::vector<vk::CommandBuffer>* cached;
    std::vector<vk::CommandBuffer>* in_use;

    if (level == vk::CommandBufferLevel::ePrimary) {
        cached = &cache_primary_cmds;
        in_use = &inuse_primary_cmds;
    } else {
        cached = &cache_secondary_cmds;
        in_use = &inuse_secondary_cmds;
    }

    if (!cached->empty()) {
        // nothing on hand... create one
        in_use->push_back(cached->back());
        cached->pop_back();
    } else {
        in_use->emplace_back();
        const vk::CommandBufferAllocateInfo info{pool, level, 1};
        check_result(context->device.allocateCommandBuffers(&info, &in_use->back()),
                     "could not allocate command buffer");
    }

    if (begin) {
        const vk::CommandBufferBeginInfo info{flags, pInheritanceInfo};
        in_use->back().begin(info);
    }

    return in_use->back();
}

std::vector<vk::CommandBuffer>
CommandPool::create_multiple(const vk::CommandBufferLevel level,
                             const uint32_t count,
                             const bool begin,
                             const vk::CommandBufferUsageFlags flags,
                             const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {

    std::vector<vk::CommandBuffer> cmds;
    std::vector<vk::CommandBuffer>* cached;
    std::vector<vk::CommandBuffer>* in_use;

    if (level == vk::CommandBufferLevel::ePrimary) {
        cached = &cache_primary_cmds;
        in_use = &inuse_primary_cmds;
    } else {
        cached = &cache_secondary_cmds;
        in_use = &inuse_secondary_cmds;
    }

    uint32_t remaining_cmds = count;
    while (!cached->empty() && remaining_cmds > 0) {
        in_use->push_back(cached->back());
        cmds.push_back(cached->back());
        cached->pop_back();
        remaining_cmds--;
    }

    if (remaining_cmds > 0) {
        const vk::CommandBufferAllocateInfo info{pool, level, remaining_cmds};
        const std::vector<vk::CommandBuffer> allocated =
            context->device.allocateCommandBuffers(info);
        insert_all(*in_use, allocated);
        move_all(cmds, allocated);
    }

    if (begin) {
        const vk::CommandBufferBeginInfo info{flags, pInheritanceInfo};
        for (vk::CommandBuffer& cmd : cmds) {
            cmd.begin(info);
        }
    }

    assert(cmds.size() == count);
    return cmds;
}

uint32_t CommandPool::get_queue_family_index() const noexcept {
    return queue_family_index;
}

const vk::CommandPool& CommandPool::get_pool() const {
    return pool;
}

void update_cached_cmds(const ContextHandle context,
                        const vk::CommandPool& pool,
                        std::vector<vk::CommandBuffer>& inuse,
                        std::vector<vk::CommandBuffer>& cached,
                        uint32_t& last_used_count) {

    // often even odd processing, therefore check last iteration too.
    const uint32_t keep_count = std::max((uint32_t)inuse.size(), last_used_count) * 1.10;
    last_used_count = (uint32_t)inuse.size();

    if (cached.size() > keep_count) {
        context->device.freeCommandBuffers(pool, cached.size() - keep_count,
                                           cached.data() + keep_count);
        cached.resize(keep_count);

        // free the rest and return
        context->device.freeCommandBuffers(pool, inuse);
        inuse.clear();
        return;
    }

    while (cached.size() < keep_count && !inuse.empty()) {
        cached.push_back(std::move(inuse.back()));
        inuse.pop_back();
    }

    if (!inuse.empty()) {
        context->device.freeCommandBuffers(pool, inuse);
        inuse.clear();
    }
}

// Frees command buffers, resets command pool
void CommandPool::reset() {
    update_cached_cmds(context, pool, inuse_primary_cmds, cache_primary_cmds,
                       last_used_primary_count);
    update_cached_cmds(context, pool, inuse_secondary_cmds, cache_secondary_cmds,
                       last_used_secondary_count);

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
