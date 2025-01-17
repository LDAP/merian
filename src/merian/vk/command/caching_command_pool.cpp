#include "merian/vk/command/caching_command_pool.hpp"
#include "merian/vk/command/command_buffer.hpp"

namespace {

void update_cached_cmds(std::vector<merian::CommandBufferHandle>& inuse,
                        std::vector<merian::CommandBufferHandle>& cached,
                        uint32_t& last_used_count) {

    // often even odd processing, therefore check last iteration too.
    const uint32_t keep_count =
        (uint32_t)(std::max((uint32_t)inuse.size(), last_used_count) * 1.10);
    last_used_count = (uint32_t)inuse.size();

    if (cached.size() > keep_count) {
        cached.resize(keep_count);
        inuse.clear();
        return;
    }

    while (cached.size() < keep_count && !inuse.empty()) {
        cached.push_back(std::move(inuse.back()));
        inuse.pop_back();
    }

    if (!inuse.empty()) {
        inuse.clear();
    }
}

} // namespace

namespace merian {

CachingCommandPool::CachingCommandPool(const CommandPoolHandle& pool)
    : CommandPool(pool->get_context()), pool(pool) {}

CachingCommandPool::~CachingCommandPool() {}

CommandBufferHandle CachingCommandPool::create(const vk::CommandBufferLevel level) {
    std::vector<CommandBufferHandle>* cached;
    std::vector<CommandBufferHandle>* in_use;

    if (level == vk::CommandBufferLevel::ePrimary) {
        cached = &cache_primary_cmds;
        in_use = &inuse_primary_cmds;
    } else {
        cached = &cache_secondary_cmds;
        in_use = &inuse_secondary_cmds;
    }

    if (!cached->empty()) {
        in_use->push_back(std::move(cached->back()));
        cached->pop_back();
    } else {
        in_use->emplace_back(std::make_shared<CommandBuffer>(pool, level));
    }

    return in_use->back();
}

[[nodiscard]]
CommandBufferHandle
CachingCommandPool::create_and_begin(const vk::CommandBufferLevel level,
                                     const vk::CommandBufferUsageFlags flags,
                                     const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    const CommandBufferHandle cmd = create(level);
    cmd->begin(flags, pInheritanceInfo);
    return cmd;
}

// ------------------------------------------------------------

const vk::CommandPool& CachingCommandPool::get_pool() const noexcept {
    return *pool;
}

CachingCommandPool::operator const vk::CommandPool&() const noexcept {
    return *pool;
}

const vk::CommandPool& CachingCommandPool::operator*() const noexcept {
    return *pool;
}

uint32_t CachingCommandPool::get_queue_family_index() const noexcept {
    return pool->get_queue_family_index();
}

// ------------------------------------------------------------

void CachingCommandPool::reset() {
    update_cached_cmds(inuse_primary_cmds, cache_primary_cmds, last_used_primary_count);
    update_cached_cmds(inuse_secondary_cmds, cache_secondary_cmds, last_used_secondary_count);

    pool->reset();
}

} // namespace merian
