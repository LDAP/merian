#pragma once

#include <memory>

#include "merian/vk/context.hpp"

namespace merian {

class RenderPass;
using RenderPassHandle = std::shared_ptr<RenderPass>;

class RenderPass : public std::enable_shared_from_this<RenderPass>, public Object {

  public:
    RenderPass(const ContextHandle& context,
               const vk::RenderPassCreateInfo2 renderpass_create_info);

    RenderPass(const ContextHandle& context, const vk::RenderPassCreateInfo renderpass_create_info);

    ~RenderPass();

    operator const vk::RenderPass&() const;

    const vk::RenderPass& get_renderpass() const;

    const vk::RenderPass& operator*() const;

    const uint32_t& get_attachment_count() const;

  private:
    const ContextHandle context;
    vk::RenderPass renderpass;
    const uint32_t attachment_count;

  public:
    static RenderPassHandle create(const ContextHandle& context,
                                   const vk::RenderPassCreateInfo2 renderpass_create_info) {
        return std::make_shared<RenderPass>(context, renderpass_create_info);
    }

    static RenderPassHandle create(const ContextHandle& context,
                                   const vk::RenderPassCreateInfo renderpass_create_info) {
        return std::make_shared<RenderPass>(context, renderpass_create_info);
    }
};

} // namespace merian
