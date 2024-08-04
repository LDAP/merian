#pragma once

#include "renderpass.hpp"

namespace merian {

class RenderpassBuilder {
  public:
    using AttachmentHandle = uint32_t;
    using SubpassHandle = uint32_t;
    static const AttachmentHandle NULL_ATTACHMENT_HANDLE = ~0;

    RenderpassBuilder(
        vk::PipelineBindPoint first_subpass_pipeline_bind_point = vk::PipelineBindPoint::eGraphics,
        const vk::SubpassDescriptionFlags first_subpass_flags = {},
        const vk::RenderPassCreateFlags renderpass_create_flags = {});

    // --- 1. Set attachments ---

    [[nodiscard]]
    AttachmentHandle
    add_attachment(const vk::ImageLayout initial_layout = vk::ImageLayout::eUndefined,
                   const vk::ImageLayout final_layout = vk::ImageLayout::eAttachmentOptimal,
                   const vk::Format format = vk::Format::eR16G16B16A16Sfloat,
                   const vk::AttachmentLoadOp load_op = vk::AttachmentLoadOp::eLoad,
                   const vk::AttachmentStoreOp store_op = vk::AttachmentStoreOp::eStore,
                   const vk::AttachmentLoadOp stencil_load_op = vk::AttachmentLoadOp::eDontCare,
                   const vk::AttachmentStoreOp stencil_store_op = vk::AttachmentStoreOp::eDontCare,
                   const vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1,
                   const vk::AttachmentDescriptionFlags flags = {});

    // --- 2. Configure subpasses ---

    [[nodiscard]]
    SubpassHandle first_subpass() const;

    [[nodiscard]]
    SubpassHandle current_subpass() const;

    SubpassHandle next_subpass(
        const vk::PipelineBindPoint subpass_pipeline_bind_point = vk::PipelineBindPoint::eGraphics,
        const vk::SubpassDescriptionFlags subpass_flags = {});

    RenderpassBuilder&
    add_input_attachment(const AttachmentHandle& attachment,
                         const vk::ImageLayout layout = vk::ImageLayout::eColorAttachmentOptimal);

    RenderpassBuilder& add_color_attachment(
        const AttachmentHandle& attachment,
        const vk::ImageLayout layout = vk::ImageLayout::eColorAttachmentOptimal,
        const AttachmentHandle& resolve_attachment = NULL_ATTACHMENT_HANDLE,
        const vk::ImageLayout resolve_layout = vk::ImageLayout::eColorAttachmentOptimal);

    RenderpassBuilder& add_preserve_attachment(const AttachmentHandle& attachment);

    RenderpassBuilder& set_depth_stencil_attachment(
        const AttachmentHandle& attachment,
        const vk::ImageLayout layout = vk::ImageLayout::eColorAttachmentOptimal);

    // --- 3. Set subpass dependencies  ---

    RenderpassBuilder& declare_subpass_depedency(
        const SubpassHandle& src,
        const SubpassHandle& dst,
        const vk::PipelineStageFlags src_stages = vk::PipelineStageFlagBits::eAllGraphics,
        const vk::PipelineStageFlags dst_stages = vk::PipelineStageFlagBits::eAllGraphics,
        const vk::AccessFlags src_access_flags = vk::AccessFlagBits::eMemoryRead |
                                                 vk::AccessFlagBits::eMemoryWrite,
        const vk::AccessFlags dst_access_flags = vk::AccessFlagBits::eMemoryRead |
                                                 vk::AccessFlagBits::eMemoryWrite,
        const vk::DependencyFlags dependency_flags = {});

    // --- 4. Build ---

    [[nodiscard]]
    RenderPassHandle build(const ContextHandle& context);

  private:
    vk::RenderPassCreateFlags renderpass_create_flags;
    std::vector<vk::AttachmentDescription> attachments;
    std::vector<vk::SubpassDescription> subpasses;
    std::vector<vk::SubpassDependency> dependencies;

    std::vector<vk::AttachmentReference> input_attachments;
    std::vector<vk::AttachmentReference> color_attachments;
    std::vector<vk::AttachmentReference> resolve_attachments;
    std::vector<uint32_t> subpass_resolve_attachment_count;
    std::vector<vk::AttachmentReference> depth_stencil_attachment;
    std::vector<uint32_t> preserve_attachments;
};

} // namespace merian
