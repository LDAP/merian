#include "utils/vector_utils.hpp"
#include "vk/utils/check_result.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

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
  private:
    struct Entry {
        vk::CommandPool pool = VK_NULL_HANDLE;
        std::vector<vk::CommandBuffer> cmds;
    };

  public:
    RingCommandPool(RingCommandPool const&) = delete;
    RingCommandPool& operator=(RingCommandPool const&) = delete;

    RingCommandPool(vk::Device device,
                    uint32_t queue_family_index,
                    vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient)
        : device(device), queue_family_index(queue_family_index), create_flags(create_flags) {

        pools.resize(RING_SIZE);
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            vk::CommandPoolCreateInfo info{create_flags, queue_family_index};
            pools[i].pool = device.createCommandPool(info);
        }
    }

    ~RingCommandPool() {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            device.destroyCommandPool(pools[i].pool);
        }
    }

    void reset() {
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            device.destroyCommandPool(pools[i].pool);
            pools[i].cmds.clear();
        }
        for (uint32_t i = 0; i < RING_SIZE; i++) {
            vk::CommandPoolCreateInfo info{create_flags, queue_family_index};
            pools[i].pool = device.createCommandPool(info);
        }
    }

    // Like set_cycle(uint32_t cycle) but advances the cycle internally by one
    void set_cycle() {
        set_cycle(current_index + 1);
    }

    // call when cycle has changed, prior creating command buffers. Use for example current_cycle_index() from
    // RingFences. Resets old pools etc and frees command buffers.
    void set_cycle(uint32_t cycle) {
        current_index = cycle % RING_SIZE;

        Entry& entry = pools[current_index];

        if (!entry.cmds.empty()) {
            device.freeCommandBuffers(entry.pool, uint32_t(entry.cmds.size()), entry.cmds.data());
            device.resetCommandPool(entry.pool, 0);
            entry.cmds.clear();
        }
    }

    // ensure proper cycle or frame is set
    vk::CommandBuffer
    createCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
                        bool begin = false,
                        vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        return createCommandBuffers(level, 1, begin, flags, pInheritanceInfo)[0];
    }

    // ensure proper cycle or frame is set
    const std::vector<vk::CommandBuffer>
    createCommandBuffers(vk::CommandBufferLevel level,
                         uint32_t count,
                         bool begin = false,
                         vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                         const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        Entry& cycle = pools[current_index];

        vk::CommandBufferAllocateInfo info{cycle.pool, level, count};

        std::vector<vk::CommandBuffer> allocated = device.allocateCommandBuffers(info);
        insert_all(cycle.cmds, allocated);

        if (begin) {
            for (vk::CommandBuffer& cmd : allocated) {
                vk::CommandBufferBeginInfo info{flags, pInheritanceInfo};
                cmd.begin(info);
            }
        }

        return allocated;
    }

  private:
    vk::Device device = VK_NULL_HANDLE;
    uint32_t queue_family_index;
    vk::CommandPoolCreateFlags create_flags;

    std::vector<Entry> pools;
    uint32_t current_index = 0;
};

} // namespace merian
