#include "renderpass_builder.hpp"

namespace merian {

RenderpassBuilder::RenderpassBuilder(const vk::SubpassDescriptionFlags first_subpass_flags,
                                     vk::PipelineBindPoint first_subpass_pipeline_bind_point) {}

uint32_t RenderpassBuilder::add_attachment(vk::ImageLayout initialLayout,
                                           vk::ImageLayout finalLayout,
                                           vk::Format format,
                                           vk::AttachmentLoadOp loadOp,
                                           vk::AttachmentStoreOp storeOp,
                                           vk::AttachmentLoadOp stencilLoadOp,
                                           vk::AttachmentStoreOp stencilStoreOp,
                                           vk::SampleCountFlagBits samples,
                                           vk::AttachmentDescriptionFlags flags) {
    const uint32_t attachment_index = attachments.size();
    attachments.emplace_back(vk::AttachmentDescription{
        flags,
        format,
        samples,
        loadOp,
        storeOp,
        stencilLoadOp,
        stencilStoreOp,
        initialLayout,
        finalLayout,
    });
    return attachment_index;
}

RenderPassHandle RenderpassBuilder::build(const ContextHandle& context) {
    // set pointers of all attachments (because they change when the vector is resized!)

    const vk::RenderPassCreateInfo create_info{renderpass_flags, attachments};

    return std::make_shared<RenderPass>(context, create_info);
}

void RenderpassBuilder::clear() {
    attachments.clear();
}

} // namespace merian
