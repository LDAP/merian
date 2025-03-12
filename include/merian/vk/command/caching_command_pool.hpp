#pragma once

#include "merian/vk/command/command_pool.hpp"

namespace merian {

class CachingCommandPool : public CommandPool {

  public:
    CachingCommandPool(const CommandPoolHandle& pool);

    ~CachingCommandPool();

    // ------------------------------------------------------------

  public:
    [[nodiscard]]
    CommandBufferHandle
    create(const vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary);

    [[nodiscard]]
    CommandBufferHandle create_and_begin(
        const vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
        const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr);

    // ------------------------------------------------------------

    const vk::CommandPool& get_pool() const noexcept override;

    operator const vk::CommandPool&() const noexcept override;

    const vk::CommandPool& operator*() const noexcept override;

    uint32_t get_queue_family_index() const noexcept override;

    // ------------------------------------------------------------

    void reset() override;

    void keep_until_pool_reset(const ObjectHandle& object) override;

    // ------------------------------------------------------------

  private:
    const CommandPoolHandle pool;

    // Estimate the necessary amount of command buffers and keep them cached
    uint32_t last_used_primary_count;
    uint32_t last_used_secondary_count;

    // Keep all cmds for resetting / freeing
    std::vector<CommandBufferHandle> inuse_primary_cmds;
    std::vector<CommandBufferHandle> inuse_secondary_cmds;

    // Keep some cmd to prevent reallocation
    std::vector<CommandBufferHandle> cache_primary_cmds;
    std::vector<CommandBufferHandle> cache_secondary_cmds;
};

} // namespace merian
