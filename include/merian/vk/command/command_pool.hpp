#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/object.hpp"
#include <vulkan/vulkan.hpp>

namespace merian {

class CommandPool : public std::enable_shared_from_this<CommandPool> {

  protected:
    // for Caching Command Pool; does not create a command pool
    CommandPool(const ContextHandle& context);

  public:
    CommandPool() = delete;

    // Cache size is the number of primary command buffers that are kept on hand to prevent
    // reallocation.
    CommandPool(
        const QueueHandle& queue,
        const vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient);

    // Cache size is the number of primary command buffers that are kept on hand to prevent
    // reallocation.
    CommandPool(
        const ContextHandle& context,
        const uint32_t queue_family_index,
        const vk::CommandPoolCreateFlags create_flags = vk::CommandPoolCreateFlagBits::eTransient);

    virtual ~CommandPool();

    // ------------------------------------------------------------

    virtual uint32_t get_queue_family_index() const noexcept;

    virtual const vk::CommandPool& get_pool() const noexcept;

    virtual operator const vk::CommandPool&() const noexcept;

    virtual const vk::CommandPool& operator*() const noexcept;

    // ------------------------------------------------------------

    // Resets the command pool and releases objects attached to this pool.
    virtual void reset();

    virtual void keep_until_pool_reset(const ObjectHandle& object) {
        objects_in_use.emplace_back(object);
    }

    virtual void keep_until_pool_reset(ObjectHandle&& object) {
        objects_in_use.emplace_back(std::move(object));
    }

    // ------------------------------------------------------------

    const ContextHandle& get_context() {
        return context;
    }

    const std::vector<ObjectHandle>& get_objects_in_use() const {
        return objects_in_use;
    }

  private:
    const ContextHandle context;
    const uint32_t queue_family_index;
    vk::CommandPool pool = VK_NULL_HANDLE;

    std::vector<ObjectHandle> objects_in_use;
};

using CommandPoolHandle = std::shared_ptr<CommandPool>;

} // namespace merian
