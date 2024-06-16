#pragma once

#include "merian/vk/context.hpp"
#include <vulkan/vulkan.hpp>

namespace merian {

class CommandPool : public std::enable_shared_from_this<CommandPool> {

  public:
    CommandPool() = delete;

    // Cache size is the number of primary command buffers that are kept on hand to prevent
    // reallocation.
    CommandPool(
        const std::shared_ptr<Queue> queue,
        const vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient);

    // Cache size is the number of primary command buffers that are kept on hand to prevent
    // reallocation.
    CommandPool(
        const SharedContext context,
        const uint32_t queue_family_index,
        const vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient);

    virtual ~CommandPool();

    operator const vk::CommandPool&() const {
        return pool;
    }

    virtual vk::CommandBuffer
    create(const vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
           const bool begin = false,
           const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
           const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr);

    virtual vk::CommandBuffer create_and_begin(
        const vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
        const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        return create(level, true, flags, pInheritanceInfo);
    }

    virtual std::vector<vk::CommandBuffer> create_multiple(
        const vk::CommandBufferLevel level,
        const uint32_t count,
        const bool begin = false,
        const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr);

    virtual std::vector<vk::CommandBuffer> create_and_begin_multiple(
        const vk::CommandBufferLevel level,
        const uint32_t count,
        const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        return create_multiple(level, count, true, flags, pInheritanceInfo);
    }

    uint32_t get_queue_family_index() const noexcept;

    const vk::CommandPool& get_pool() const;

    // Frees all primary command buffers that were allocated from this pool and resets command pool
    // (keeps some primaries in cache)
    void reset();

    // Ends all command primary command buffers buffers
    void end_all();

    // Returns all primary command buffers that were allocated after the last reset()
    const std::vector<vk::CommandBuffer>& get_command_buffers() const;

  private:
    const SharedContext context;
    const uint32_t queue_family_index;
    vk::CommandPool pool = VK_NULL_HANDLE;

    // Estimate the necessary amount of command buffers and keep them cached
    uint32_t last_used_primary_count;
    uint32_t last_used_secondary_count;

    // Keep all cmds for resetting / freeing
    std::vector<vk::CommandBuffer> inuse_primary_cmds;
    std::vector<vk::CommandBuffer> inuse_secondary_cmds;

    // Keep some cmd to prevent reallocation
    std::vector<vk::CommandBuffer> cache_primary_cmds;
    std::vector<vk::CommandBuffer> cache_secondary_cmds;
};

using CommandPoolHandle = std::shared_ptr<CommandPool>;

} // namespace merian
