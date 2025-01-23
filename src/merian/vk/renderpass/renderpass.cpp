#include "merian/vk/renderpass/renderpass.hpp"

namespace merian {

RenderPass::RenderPass(const ContextHandle& context,
                       const vk::RenderPassCreateInfo2 renderpass_create_info)
    : context(context), attachment_count(renderpass_create_info.attachmentCount) {

    renderpass = context->device.createRenderPass2(renderpass_create_info);
    SPDLOG_DEBUG("create renderpass ({})", fmt::ptr(static_cast<VkRenderPass>(renderpass)));
}

RenderPass::RenderPass(const ContextHandle& context,
                       const vk::RenderPassCreateInfo renderpass_create_info)
    : context(context), attachment_count(renderpass_create_info.attachmentCount) {

    renderpass = context->device.createRenderPass(renderpass_create_info);
    SPDLOG_DEBUG("create renderpass ({})", fmt::ptr(static_cast<VkRenderPass>(renderpass)));
}

RenderPass::~RenderPass() {
    SPDLOG_DEBUG("destroy renderpass ({})", fmt::ptr(static_cast<VkRenderPass>(renderpass)));

    context->device.destroyRenderPass(renderpass);
}

RenderPass::operator const vk::RenderPass&() const {
    return renderpass;
}

const vk::RenderPass& RenderPass::get_renderpass() const {
    return renderpass;
}

const vk::RenderPass& RenderPass::operator*() const {
    return renderpass;
}

const uint32_t& RenderPass::get_attachment_count() const {
    return attachment_count;
}

} // namespace merian
