#pragma once

#include "merian/vk/context.hpp"

#include <memory>
#include <spdlog/spdlog.h>

namespace merian {

class Surface : public std::enable_shared_from_this<Surface> {

  public:
    // Manage the supplied surface.
    Surface(const ContextHandle& context, const vk::SurfaceKHR& surface)
        : context(context), surface(surface),
          capabilities(
              context->get_physical_device()->get_physical_device().getSurfaceCapabilitiesKHR(
                  surface)) {
        SPDLOG_DEBUG("create surface ({})", fmt::ptr(this));
    }

    ~Surface() {
        SPDLOG_DEBUG("destroy surface ({})", fmt::ptr(this));
        context->get_instance()->get_instance().destroySurfaceKHR(surface);
    }

    operator const vk::SurfaceKHR&() const {
        return surface;
    }

    const vk::SurfaceKHR& get_surface() const {
        return surface;
    }

    const vk::SurfaceCapabilitiesKHR& get_capabilities() const {
        return capabilities;
    }

  private:
    const ContextHandle context;
    vk::SurfaceKHR surface;
    vk::SurfaceCapabilitiesKHR capabilities;
};

using SurfaceHandle = std::shared_ptr<Surface>;

} // namespace merian
