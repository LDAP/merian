#include "merian/vk/renderpass/renderpass_builder.hpp"

namespace merian {

RenderpassBuilder::RenderpassBuilder(vk::PipelineBindPoint first_subpass_pipeline_bind_point,
                                     const vk::SubpassDescriptionFlags first_subpass_flags,
                                     const vk::RenderPassCreateFlags renderpass_create_flags)
    : renderpass_create_flags(renderpass_create_flags) {
    std::ignore = next_subpass(first_subpass_pipeline_bind_point, first_subpass_flags);
}

// --- 1. Set attachments ---

// returns the attachment index
RenderpassBuilder::AttachmentHandle
RenderpassBuilder::add_attachment(const vk::ImageLayout initial_layout,
                                  const vk::ImageLayout final_layout,
                                  const vk::Format format,
                                  const vk::AttachmentLoadOp load_op,
                                  const vk::AttachmentStoreOp store_op,
                                  const vk::AttachmentLoadOp stencil_load_op,
                                  const vk::AttachmentStoreOp stencil_store_op,
                                  const vk::SampleCountFlagBits samples,
                                  const vk::AttachmentDescriptionFlags flags) {
    const uint32_t attachment_index = attachments.size();
    attachments.emplace_back(vk::AttachmentDescription{
        flags,
        format,
        samples,
        load_op,
        store_op,
        stencil_load_op,
        stencil_store_op,
        initial_layout,
        final_layout,
    });
    return attachment_index;
}

// --- 2. Configure subpasses ---

RenderpassBuilder::SubpassHandle RenderpassBuilder::first_subpass() const {
    return 0;
}

RenderpassBuilder::SubpassHandle RenderpassBuilder::current_subpass() const {
    return subpasses.size() - 1;
}

// Starts a new subpass and returns its handle
RenderpassBuilder::SubpassHandle
RenderpassBuilder::next_subpass(const vk::PipelineBindPoint subpass_pipeline_bind_point,
                                const vk::SubpassDescriptionFlags subpass_flags) {
    subpasses.emplace_back(vk::SubpassDescription{
        subpass_flags,
        subpass_pipeline_bind_point,
        0,
        nullptr,
        0,
        nullptr,
        nullptr,
        nullptr,
        0,
        nullptr,
    });
    subpass_resolve_attachment_count.emplace_back(0);
    depth_stencil_attachment.emplace_back(
        vk::AttachmentReference{VK_ATTACHMENT_UNUSED, vk::ImageLayout::eUndefined});

    return current_subpass();
}

RenderpassBuilder& RenderpassBuilder::add_input_attachment(const AttachmentHandle& attachment,
                                                           const vk::ImageLayout layout) {
    assert(attachment < attachments.size());
    input_attachments.emplace_back(vk::AttachmentReference{attachment, layout});
    subpasses.back().inputAttachmentCount++;
    return *this;
}

RenderpassBuilder&
RenderpassBuilder::add_color_attachment(const AttachmentHandle& attachment,
                                        const vk::ImageLayout layout,
                                        const AttachmentHandle& resolve_attachment,
                                        const vk::ImageLayout resolve_layout) {
    assert(attachment < attachments.size());
    color_attachments.emplace_back(vk::AttachmentReference{attachment, layout});
    subpasses.back().colorAttachmentCount++;

    if (resolve_attachment != NULL_ATTACHMENT_HANDLE) {
        assert(resolve_attachment < attachments.size());
        resolve_attachments.emplace_back(
            vk::AttachmentReference{resolve_attachment, resolve_layout});
        subpass_resolve_attachment_count.back()++;
    }

    assert((subpass_resolve_attachment_count.back() == 0 ||
            subpasses.back().colorAttachmentCount == subpass_resolve_attachment_count.back()) &&
           "resolve attachment count must be 0 or equal to color attachment count");
    return *this;
}

RenderpassBuilder& RenderpassBuilder::add_preserve_attachment(const AttachmentHandle& attachment) {
    assert(attachment < attachments.size());
    preserve_attachments.push_back(attachment);
    subpasses.back().preserveAttachmentCount++;
    return *this;
}

RenderpassBuilder&
RenderpassBuilder::set_depth_stencil_attachment(const AttachmentHandle& attachment,
                                                const vk::ImageLayout layout) {
    assert(attachment < attachments.size());
    depth_stencil_attachment.back() = vk::AttachmentReference{attachment, layout};
    return *this;
}

// --- 3. Set subpass dependencies  ---

RenderpassBuilder&
RenderpassBuilder::declare_subpass_depedency(const SubpassHandle& src,
                                             const SubpassHandle& dst,
                                             const vk::PipelineStageFlags src_stages,
                                             const vk::PipelineStageFlags dst_stages,
                                             const vk::AccessFlags src_access_flags,
                                             const vk::AccessFlags dst_access_flags,
                                             const vk::DependencyFlags dependency_flags) {
    assert(src < subpasses.size());
    assert(dst < subpasses.size());

    dependencies.emplace_back(vk::SubpassDependency{
        src,
        dst,
        src_stages,
        dst_stages,
        src_access_flags,
        dst_access_flags,
        dependency_flags,
    });

    return *this;
}

// --- 4. Build ---

RenderPassHandle RenderpassBuilder::build(const ContextHandle& context) {
    assert(!subpasses.empty());
    assert(subpasses.size() == subpass_resolve_attachment_count.size());
    assert(subpasses.size() == depth_stencil_attachment.size());

    vk::AttachmentReference* p_input_attachments = input_attachments.data();
    vk::AttachmentReference* p_color_attachments = color_attachments.data();
    vk::AttachmentReference* p_resolve_attachments = resolve_attachments.data();
    uint32_t* p_preserve_attachments = preserve_attachments.data();

    // set pointers of all attachment references (because they change when the vector is resized!)
    for (uint32_t i = 0; i < subpasses.size(); i++) {
        vk::SubpassDescription& subpass = subpasses[i];

        subpass.pInputAttachments = p_input_attachments;
        p_input_attachments += subpass.inputAttachmentCount;
        subpass.pColorAttachments = p_color_attachments;
        p_color_attachments += subpass.colorAttachmentCount;
        subpass.pPreserveAttachments = p_preserve_attachments;
        p_preserve_attachments += subpass.preserveAttachmentCount;
        subpass.pDepthStencilAttachment = &depth_stencil_attachment[i];

        if (subpass_resolve_attachment_count[i] > 0) {
            assert(subpass.colorAttachmentCount == subpass_resolve_attachment_count[i]);
            subpass.pResolveAttachments = p_resolve_attachments;
            p_resolve_attachments += subpass.colorAttachmentCount;
        }
    }

    const vk::RenderPassCreateInfo create_info{
        renderpass_create_flags,
        attachments,
        subpasses,
        dependencies,
    };
    return std::make_shared<RenderPass>(context, create_info);
}

} // namespace merian
