#pragma once

#include "merian/vk/context.hpp"

#include <memory>
#include <spdlog/spdlog.h>

namespace merian {

class Surface : public std::enable_shared_from_this<Surface> {

  public:
    // Manage the supplied surface.
    Surface(const ContextHandle& context, const vk::SurfaceKHR& surface)
        : context(context), surface(surface) {
        SPDLOG_DEBUG("create surface ({})", fmt::ptr(this));
    }

    ~Surface() {
        SPDLOG_DEBUG("destroy surface ({})", fmt::ptr(this));
        context->instance.destroySurfaceKHR(surface);
    }

    operator const vk::SurfaceKHR&() const {
        return surface;
    }

    const vk::SurfaceKHR& get_surface() const {
        return surface;
    }

  private:
    const ContextHandle context;
    vk::SurfaceKHR surface;
};

using SurfaceHandle = std::shared_ptr<Surface>;

} // namespace merian
