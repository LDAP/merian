#pragma once

#include "merian/utils/vector.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/utils/check_result.hpp"
#include <vulkan/vulkan.hpp>

namespace merian {

class CommandPool : public std::enable_shared_from_this<CommandPool> {

  public:
    CommandPool() = delete;

    // Cache size is the number of primary command buffers that are kept on hand to prevent
    // reallocation.
    CommandPool(const std::shared_ptr<Queue> queue,
                vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient,
                const uint32_t cache_size = 10);

    // Cache size is the number of primary command buffers that are kept on hand to prevent
    // reallocation.
    CommandPool(const SharedContext context,
                uint32_t queue_family_index,
                vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient,
                const uint32_t cache_size = 10);

    virtual ~CommandPool();

    operator const vk::CommandPool&() const {
        return pool;
    }

    virtual vk::CommandBuffer
    create(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
           bool begin = false,
           vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
           const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr);

    virtual vk::CommandBuffer create_and_begin(
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
        vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        return create(level, true, flags, pInheritanceInfo);
    }

    virtual std::vector<vk::CommandBuffer> create_multiple(
        vk::CommandBufferLevel level,
        uint32_t count,
        bool begin = false,
        vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr);

    virtual std::vector<vk::CommandBuffer> create_and_begin_multiple(
        vk::CommandBufferLevel level,
        uint32_t count,
        vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        return create_multiple(level, count, true, flags, pInheritanceInfo);
    }

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

  private:
    vk::CommandPool pool = VK_NULL_HANDLE;

    std::vector<vk::CommandBuffer> inuse_primary_cmds;
    // Keep some cmd to prevent reallocation
    std::vector<vk::CommandBuffer> cache_primary_cmds;

    const uint32_t cache_size;
};

using CommandPoolHandle = std::shared_ptr<CommandPool>;

} // namespace merian
