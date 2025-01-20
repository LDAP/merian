#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/renderpass/renderpass.hpp"

namespace merian {

class Framebuffer;
using FramebufferHandle = std::shared_ptr<Framebuffer>;

class Framebuffer : public Object {

  public:
    Framebuffer(const ContextHandle& context,
                const RenderPassHandle& renderpass,
                const uint32_t width,
                const uint32_t height,
                const uint32_t layers = 1,
                const std::vector<vk::ImageView>& attachments = {},
                const vk::FramebufferCreateFlags flags = {})
        : context(context), renderpass(renderpass), extent(width, height) {
        assert(attachments.size() == renderpass->get_attachment_count());
        vk::FramebufferCreateInfo create_info{flags, *renderpass, attachments,
                                              width, height,      layers};
        framebuffer = context->device.createFramebuffer(create_info);
    }

    ~Framebuffer() {
        context->device.destroyFramebuffer(framebuffer);
    }

    operator const vk::Framebuffer&() const {
        return framebuffer;
    }

    const vk::Framebuffer& operator*() const {
        return framebuffer;
    }

    const vk::Framebuffer& get() const {
        return framebuffer;
    }

    const RenderPassHandle& get_renderpass() const {
        return renderpass;
    }

    const vk::Extent2D& get_extent() const {
        return extent;
    }

  private:
    const ContextHandle context;
    const RenderPassHandle renderpass;
    vk::Extent2D extent;
    vk::Framebuffer framebuffer;

  public:
    static FramebufferHandle create(const ContextHandle& context,
                                    const RenderPassHandle& renderpass,
                                    const uint32_t width,
                                    const uint32_t height,
                                    const uint32_t layers = 1,
                                    const std::vector<vk::ImageView>& attachments = {},
                                    const vk::FramebufferCreateFlags flags = {}) {
        return std::make_shared<Framebuffer>(context, renderpass, width, height, layers,
                                             attachments, flags);
    }

    static FramebufferHandle create(const ContextHandle& context,
                                    const RenderPassHandle& renderpass,
                                    const vk::Extent3D extent,
                                    const std::vector<vk::ImageView>& attachments = {},
                                    const vk::FramebufferCreateFlags flags = {}) {
        return std::make_shared<Framebuffer>(context, renderpass, extent.width, extent.height,
                                             extent.depth, attachments, flags);
    }
};

} // namespace merian
