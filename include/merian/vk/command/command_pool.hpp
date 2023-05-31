#pragma once

#include "merian/utils/vector_utils.hpp"
#include <vulkan/vulkan.hpp>

// Forward def
class QueueContainer;

namespace merian {

class CommandPool {

  private:
    // Can submit all cmds as batch
    friend class QueueContainer;

  public:
    CommandPool() = delete;

    CommandPool(vk::Device& device,
                uint32_t queue_family_index,
                vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient)
        : device(device) {
        vk::CommandPoolCreateInfo info{create_flags, queue_family_index};
        pool = device.createCommandPool(info);
    };

    virtual ~CommandPool(){
        reset();
        device.destroyCommandPool(pool);
    };

    virtual vk::CommandBuffer
    createCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
                        bool begin = true,
                        vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        return createCommandBuffers(level, 1, begin, flags, pInheritanceInfo)[0];
    }

    virtual std::vector<vk::CommandBuffer>
    createCommandBuffers(vk::CommandBufferLevel level,
                         uint32_t count,
                         bool begin = true,
                         vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                         const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        vk::CommandBufferAllocateInfo info{pool, level, count};

        std::vector<vk::CommandBuffer> allocated = device.allocateCommandBuffers(info);
        insert_all(cmds, allocated);

        if (begin) {
            for (vk::CommandBuffer& cmd : allocated) {
                vk::CommandBufferBeginInfo info{flags, pInheritanceInfo};
                cmd.begin(info);
            }
        }

        return allocated;
    }

    // Frees command buffers, resets command pool
    void reset() {
        device.freeCommandBuffers(pool, cmds);
        device.resetCommandPool(pool);
        cmds.clear();
    }

    bool has_cmds() {
        return !cmds.empty();
    }

  private:
    vk::Device& device;

  private:
    vk::CommandPool pool = VK_NULL_HANDLE;
    std::vector<vk::CommandBuffer> cmds;
};

} // namespace merian
