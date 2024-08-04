#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/renderpass/renderpass.hpp"

namespace merian {

class Framebuffer {

  public:
    Framebuffer(const ContextHandle& context,
                const RenderPassHandle& renderpass,
                const uint32_t width,
                const uint32_t height,
                const uint32_t layers = 1,
                const std::vector<vk::ImageView>& attachments = {},
                const vk::FramebufferCreateFlags flags = {})
        : context(context), renderpass(renderpass), extent(width, height) {
        vk::FramebufferCreateInfo create_info{flags, *renderpass, attachments,
                                              width, height,      layers};
        framebuffer = context->device.createFramebuffer(create_info);
    }

    Framebuffer(const ContextHandle& context,
                const RenderPassHandle& renderpass,
                const vk::Extent3D extent,
                const std::vector<vk::ImageView>& attachments = {},
                const vk::FramebufferCreateFlags flags = {})
        : merian::Framebuffer(
              context, renderpass, extent.width, extent.height, extent.depth, attachments, flags) {}

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

    void
    begin_renderpass(const vk::CommandBuffer& cmd,
                     const vk::Rect2D& render_area,
                     const std::vector<vk::ClearValue>& clear_values = {},
                     const vk::SubpassContents subpass_contents = vk::SubpassContents::eInline) {
        vk::RenderPassBeginInfo begin_info{*renderpass, framebuffer, render_area, clear_values};
        cmd.beginRenderPass(begin_info, subpass_contents);
    }

    void
    begin_renderpass(const vk::CommandBuffer& cmd,
                     const std::vector<vk::ClearValue>& clear_values = {},
                     const vk::SubpassContents subpass_contents = vk::SubpassContents::eInline) {
        begin_renderpass(cmd, vk::Rect2D{{}, extent}, clear_values, subpass_contents);
    }

  private:
    const ContextHandle context;
    const RenderPassHandle renderpass;
    vk::Extent2D extent;
    vk::Framebuffer framebuffer;
};

using FramebufferHandle = std::shared_ptr<Framebuffer>;

} // namespace merian
