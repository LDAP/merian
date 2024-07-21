#include "renderpass.hpp"

namespace merian {

RenderPass::RenderPass(const ContextHandle context,
                       const vk::RenderPassCreateInfo2 renderpass_create_info)
    : context(context) {
    renderpass = context->device.createRenderPass2(renderpass_create_info);
}

RenderPass::RenderPass(const ContextHandle context,
                       const vk::RenderPassCreateInfo renderpass_create_info)
    : context(context) {
    renderpass = context->device.createRenderPass(renderpass_create_info);
}

RenderPass::~RenderPass() {
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

} // namespace merian
