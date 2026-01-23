#pragma once

#include "merian/vk/context.hpp"

namespace merian {

class Fence;
using FenceHandle = std::shared_ptr<Fence>;

class Fence : public std::enable_shared_from_this<Fence> {
  private:
    Fence(const ContextHandle& context, const vk::FenceCreateFlags flags) : context(context) {
        fence = context->get_device()->get_device().createFence(vk::FenceCreateInfo{flags});
    }

  public:
    ~Fence() {
        context->get_device()->get_device().destroyFence(fence);
    }

    // -----------------------------------------------------------------

    // Returns false if the timeout passes without the fence being signaled.
    bool wait(const uint64_t timeout = ~0) {
        return context->get_device()->get_device().waitForFences(fence, VK_TRUE, timeout) == vk::Result::eSuccess;
    }

    bool is_signaled() {
        return context->get_device()->get_device().waitForFences(fence, VK_TRUE, 0) == vk::Result::eSuccess;
    }

    void reset() {
        context->get_device()->get_device().resetFences(fence);
    }

    // -----------------------------------------------------------------

    const vk::Fence& get_fence() {
        return fence;
    }

    operator const vk::Fence&() {
        return fence;
    }

    const vk::Fence& operator*() {
        return fence;
    }

  private:
    const ContextHandle context;
    vk::Fence fence;

  public:
    static FenceHandle create(const ContextHandle& context, const vk::FenceCreateFlags flags = {}) {
        return std::shared_ptr<Fence>(new Fence(context, flags));
    }

    static FenceHandle create(const ContextHandle& context, const bool signaled) {
        vk::FenceCreateFlags flags;
        if (signaled) {
            flags |= vk::FenceCreateFlagBits::eSignaled;
        }
        return create(context, flags);
    }
};

} // namespace merian
