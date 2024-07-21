#pragma once

#include "renderpass.hpp"

namespace merian {

class RenderpassBuilder {
  public:
    RenderpassBuilder();

    uint32_t add_attachment(vk::AttachmentDescriptionFlags flags_ = {},
                        vk::Format format_ = vk::Format::eUndefined,
                        vk::SampleCountFlagBits samples_ = vk::SampleCountFlagBits::e1,
                        vk::AttachmentLoadOp loadOp_ = vk::AttachmentLoadOp::eLoad,
                        vk::AttachmentStoreOp storeOp_ = vk::AttachmentStoreOp::eStore,
                        vk::AttachmentLoadOp stencilLoadOp_ = vk::AttachmentLoadOp::eLoad,
                        vk::AttachmentStoreOp stencilStoreOp_ = vk::AttachmentStoreOp::eStore,
                        vk::ImageLayout initialLayout_ = vk::ImageLayout::eUndefined,
                        vk::ImageLayout finalLayout_ = vk::ImageLayout::eUndefined);

    void clear() {
        attachments.clear();
    }

  private:
    std::vector<vk::AttachmentDescription> attachments;



};

} // namespace merian
