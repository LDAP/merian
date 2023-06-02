#pragma once

#include "merian/vk/command/command_pool.hpp"
#include "merian/vk/utils/check_result.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

class RingCommandPoolCycle : public CommandPool {

  public:
    RingCommandPoolCycle() = delete;

    RingCommandPoolCycle(vk::Device& device,
                         uint32_t queue_family_index,
                         vk::CommandPoolCreateFlags create_flags,
                         uint32_t cycle_index,
                         uint32_t& current_index)
        : CommandPool(device, queue_family_index, create_flags), cycle_index(cycle_index),
          current_index(current_index){};

    vk::CommandBuffer
    create(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
                        bool begin = false,
                        vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) override {
        assert(current_index == cycle_index);
        return CommandPool::create(level, begin, flags, pInheritanceInfo);
    }

    std::vector<vk::CommandBuffer>
    create_multiple(vk::CommandBufferLevel level,
                         uint32_t count,
                         bool begin = false,
                         vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                         const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) override {
        assert(current_index == cycle_index);
        return CommandPool::create_multiple(level, count, begin, flags, pInheritanceInfo);
    }

  private:
    uint32_t cycle_index;
    uint32_t& current_index;
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
template <uint32_t RING_SIZE = 3> class RingCommandPool {
  public:
    RingCommandPool(RingCommandPool const&) = delete;
    RingCommandPool& operator=(RingCommandPool const&) = delete;

    RingCommandPool(vk::Device device,
                    uint32_t queue_family_index,
                    vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient)
        : device(device), queue_family_index(queue_family_index), create_flags(create_flags) {

        for (uint32_t i = 0; i < RING_SIZE; i++) {
            pools.emplace_back(device, queue_family_index, create_flags, i, current_index);
        }
    }

    void reset() {
        pools.clear();
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            pools.emplace_back(device, queue_family_index, create_flags, i, current_index);
        }
    }

    // Like set_cycle(uint32_t cycle) but advances the cycle internally by one
    CommandPool& set_cycle() {
        set_cycle(current_index + 1);
    }

    // call when cycle has changed, prior creating command buffers. Use for example current_cycle_index() from
    // RingFences. Resets old pools etc and frees command buffers.
    CommandPool& set_cycle(uint32_t cycle) {
        current_index = cycle % RING_SIZE;
        RingCommandPoolCycle& current_pool = pools[current_index];

        if (current_pool.has_command_buffers()) {
            current_pool.reset();
        }

        return current_pool;
    }

  private:
    vk::Device device = VK_NULL_HANDLE;
    uint32_t queue_family_index;
    vk::CommandPoolCreateFlags create_flags;

    std::vector<RingCommandPoolCycle> pools;
    uint32_t current_index = 0;
};

} // namespace merian
