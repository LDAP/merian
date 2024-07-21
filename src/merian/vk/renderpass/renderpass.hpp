#pragma once

#include <memory>

#include "merian/vk/context.hpp"

namespace merian {

class RenderPass : public std::enable_shared_from_this<RenderPass> {

  public:
    RenderPass(const ContextHandle context, const vk::RenderPassCreateInfo2 renderpass_create_info);

    RenderPass(const ContextHandle context, const vk::RenderPassCreateInfo renderpass_create_info);

    ~RenderPass();

    operator const vk::RenderPass&() const;

    const vk::RenderPass& get_renderpass() const;

    const vk::RenderPass& operator*() const;

  private:
    const ContextHandle context;
    vk::RenderPass renderpass;
};

using RenderPassHandle = std::shared_ptr<RenderPass>;

} // namespace merian
