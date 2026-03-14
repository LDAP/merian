#pragma once

#include "merian/vk/device.hpp"

#include <memory>
#include <spdlog/spdlog.h>

namespace merian {

class Surface : public std::enable_shared_from_this<Surface> {

  public:
    // Manage the supplied surface.
    Surface(const DeviceHandle& device, const vk::SurfaceKHR& surface)
        : device(device), surface(surface) {
        SPDLOG_DEBUG("create surface ({})", fmt::ptr(this));
    }

    ~Surface() {
        SPDLOG_DEBUG("destroy surface ({})", fmt::ptr(this));
        device->get_physical_device()->get_instance()->get_instance().destroySurfaceKHR(surface);
    }

    operator const vk::SurfaceKHR&() const {
        return surface;
    }

    const vk::SurfaceKHR& get_surface() const {
        return surface;
    }

    vk::SurfaceCapabilitiesKHR get_capabilities() const {
        return device->get_physical_device()->get_physical_device().getSurfaceCapabilitiesKHR(
            surface);
    }

  private:
    const DeviceHandle device;
    vk::SurfaceKHR surface;
};

using SurfaceHandle = std::shared_ptr<Surface>;

} // namespace merian
