#pragma once

#include "merian/utils/vector.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/utils/check_result.hpp"
#include <vulkan/vulkan.hpp>

namespace merian {

class CommandPool : public std::enable_shared_from_this<CommandPool> {

  public:
    CommandPool() = delete;

    CommandPool(
        const std::shared_ptr<Queue> queue,
        vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient);

    CommandPool(
        const SharedContext context,
        uint32_t queue_family_index,
        vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient);

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

    vk::CommandPool& get_pool();
    // Frees command buffers, resets command pool
    void reset();
    bool has_command_buffers();
    // Ends all command buffers
    void end_all();
    const std::vector<vk::CommandBuffer>& get_command_buffers() const;

  private:
    const SharedContext context;

  private:
    vk::CommandPool pool = VK_NULL_HANDLE;
    std::vector<vk::CommandBuffer> cmds;
};

using CommandPoolHandle = std::shared_ptr<CommandPool>;

} // namespace merian
