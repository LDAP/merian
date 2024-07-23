#pragma once

#include "renderpass.hpp"

namespace merian {

class RenderpassBuilder {
  public:
    using AttachmentHandle = uint32_t;
    using SubpassHandle = uint32_t;

    RenderpassBuilder(
        const vk::SubpassDescriptionFlags first_subpass_flags = {},
        vk::PipelineBindPoint first_subpass_pipeline_bind_point = vk::PipelineBindPoint::eGraphics);

    // --- 1. Set attachments ---

    // returns the attachment index
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
    SubpassHandle current_subpass();

    // Starts a new subpass and returns it handle
    [[nodiscard]]
    SubpassHandle
    next_subpass(const vk::SubpassDescriptionFlags subpass_flags,
                 vk::PipelineBindPoint pipelineBindPoint = vk::PipelineBindPoint::eGraphics);

    RenderpassBuilder&
    add_input_attachment(const AttachmentHandle& attachment,
                         vk::ImageLayout layout = vk::ImageLayout::eColorAttachmentOptimal);

    RenderpassBuilder&
    add_color_attachment(const AttachmentHandle& attachment,
                         vk::ImageLayout layout = vk::ImageLayout::eColorAttachmentOptimal);

    RenderpassBuilder&
    add_resolve_attachment(const AttachmentHandle& attachment,
                           vk::ImageLayout layout = vk::ImageLayout::eColorAttachmentOptimal);

    RenderpassBuilder&
    set_depth_stencil_attachment(const AttachmentHandle& attachment,
                                 vk::ImageLayout layout = vk::ImageLayout::eColorAttachmentOptimal);

    // --- 3. Set subpass dependencies  ---

    RenderpassBuilder& declare_subpass_depedency(
        const SubpassHandle& src,
        const SubpassHandle& dst,
        const vk::PipelineStageFlags src_stages = vk::PipelineStageFlagBits::eAllGraphics,
        const vk::PipelineStageFlags dst_stages = vk::PipelineStageFlagBits::eAllGraphics,
        const vk::AccessFlags src_access_flags = vk::AccessFlagBits::eMemoryRead |
                                                 vk::AccessFlagBits::eMemoryWrite,
        const vk::AccessFlags dst_access_flags = vk::AccessFlagBits::eMemoryRead |
                                                 vk::AccessFlagBits::eMemoryWrite);

    // --- 4. Build ---

    [[nodiscard]]
    RenderPassHandle build(const ContextHandle& context);

    // Resets the builder to build a new RenderPass
    void clear();

  private:
    vk::RenderPassCreateFlags renderpass_flags;
    std::vector<vk::AttachmentDescription> attachments;
    std::vector<vk::SubpassDescription> subpasses;
    std::vector<vk::SubpassDependency> dependencies;

    const std::vector<vk::AttachmentReference> input_attachments;
    const std::vector<vk::AttachmentReference> color_attachments;
    const std::vector<vk::AttachmentReference> resolve_attachments;
    const std::vector<vk::AttachmentReference> depth_stencil_attachment;
    const std::vector<uint32_t> preserve_attachments;
};

} // namespace merian
