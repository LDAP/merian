#pragma once

#include <memory>

#include "merian/vk/context.hpp"

namespace merian {

class RenderPass : public std::enable_shared_from_this<RenderPass> {

  public:
    ~RenderPass() {
        context->device.destroyRenderPass(renderpass);
    }

    operator const vk::RenderPass&() const {
        return renderpass;
    }

    const vk::RenderPass& get_renderpass() const {
        return renderpass;
    }

    const vk::RenderPass& operator*() const {
        return renderpass;
    }

  private:
    const SharedContext context;

    vk::RenderPass renderpass;
};

using RenderPassHandle = std::shared_ptr<RenderPass>;

} // namespace merian
