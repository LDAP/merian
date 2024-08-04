#pragma once

#include "merian/vk/command/command_pool.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

class RingCommandPoolCycle : public CommandPool {

  public:
    RingCommandPoolCycle() = delete;

    RingCommandPoolCycle(const ContextHandle& context,
                         const uint32_t queue_family_index,
                         const vk::CommandPoolCreateFlags create_flags,
                         const uint32_t cycle_index,
                         const uint32_t& current_index);

    vk::CommandBuffer
    create(const vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
           const bool begin = false,
           const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
           const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) override;

    vk::CommandBuffer create_and_begin(
        const vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
        const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) override;

    std::vector<vk::CommandBuffer> create_multiple(
        const vk::CommandBufferLevel level,
        const uint32_t count,
        const bool begin = false,
        const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) override;

    std::vector<vk::CommandBuffer> create_and_begin_multiple(
        const vk::CommandBufferLevel level,
        const uint32_t count,
        const vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) override;

  private:
    const uint32_t cycle_index;
    const uint32_t& current_index;
};

/**
 * @brief  manages a fixed cycle set of VkCommandBufferPools and
 * one-shot command buffers allocated from them.
 *
 * The usage of multiple command buffer pools also means we get nice allocation
 * behavior (linear allocation from frame start to frame end) without fragmentation.
 * If we were using a single command pool over multiple frames, it could fragment easily.
 *
 * You must ensure cycle is available manually, typically by keeping in sync
 * with ring fences.
 */
template <uint32_t RING_SIZE = 2> class RingCommandPool {
  public:
    RingCommandPool(RingCommandPool const&) = delete;
    RingCommandPool& operator=(RingCommandPool const&) = delete;

    RingCommandPool(
        const ContextHandle& context,
        const QueueHandle queue,
        vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient)
        : context(context), create_flags(create_flags) {
        const uint32_t queue_family_index = queue->get_queue_family_index();
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            pools[i] = std::make_shared<RingCommandPoolCycle>(context, queue_family_index,
                                                              create_flags, i, current_index);
        }
    }

    // Like set_cycle(uint32_t cycle) but advances the cycle internally by one
    CommandPoolHandle set_cycle() {
        return set_cycle(current_index + 1);
    }

    // call when cycle has changed, prior creating command buffers. Use for example
    // current_cycle_index() from RingFences. Resets old pools etc and frees command buffers.
    CommandPoolHandle set_cycle(uint32_t cycle) {
        current_index = cycle % RING_SIZE;
        std::shared_ptr<RingCommandPoolCycle>& current_pool = pools[current_index];

        current_pool->reset();

        return current_pool;
    }

  private:
    const ContextHandle context;
    vk::CommandPoolCreateFlags create_flags;

    std::array<std::shared_ptr<RingCommandPoolCycle>, RING_SIZE> pools;
    uint32_t current_index = 0;
};

template <uint32_t RING_SIZE>
using RingCommandPoolHandle = std::shared_ptr<RingCommandPool<RING_SIZE>>;

} // namespace merian
