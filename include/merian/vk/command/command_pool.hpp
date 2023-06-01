#pragma once

#include "merian/utils/vector_utils.hpp"
#include "merian/vk/utils/check_result.hpp"
#include <vulkan/vulkan.hpp>

namespace merian {

class CommandPool {

  public:
    CommandPool() = delete;

    CommandPool(vk::Device& device,
                uint32_t queue_family_index,
                vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient)
        : device(device) {
        vk::CommandPoolCreateInfo info{create_flags, queue_family_index};
        pool = device.createCommandPool(info);
    };

    virtual ~CommandPool() {
        reset();
        device.destroyCommandPool(pool);
    };

    virtual vk::CommandBuffer
    createCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
                        bool begin = true,
                        vk::CommandBufferUsageFlags flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
                        const vk::CommandBufferInheritanceInfo* pInheritanceInfo = nullptr) {
        cmds.emplace_back();
        vk::CommandBuffer& cmd = cmds[cmds.size() - 1];

        vk::CommandBufferAllocateInfo info{pool, level, 1};
        check_result(device.allocateCommandBuffers(&info, &cmd), "could not allocate command buffer");

        if (begin) {
            vk::CommandBufferBeginInfo info{flags, pInheritanceInfo};
            cmd.begin(info);
        }

        return cmd;
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
            vk::CommandBufferBeginInfo info{flags, pInheritanceInfo};
            for (vk::CommandBuffer& cmd : allocated) {
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

    bool has_command_buffers() {
        return !cmds.empty();
    }

    // Ends all command buffers
    void end() {
        for (vk::CommandBuffer& cmd : cmds) {
            cmd.end();
        }
    }

    const std::vector<vk::CommandBuffer>& get_command_buffers() const {
        return cmds;
    }

  private:
    vk::Device& device;

  private:
    vk::CommandPool pool = VK_NULL_HANDLE;
    std::vector<vk::CommandBuffer> cmds;
};

} // namespace merian
