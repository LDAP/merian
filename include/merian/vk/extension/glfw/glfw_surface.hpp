#pragma once

#include "merian/vk/extension/glfw/glfw_window.hpp"
#include "merian/vk/window/surface.hpp"

#include <spdlog/spdlog.h>

namespace merian {

inline vk::SurfaceKHR surface_from_glfw_window(const DeviceHandle& device,
                                               const GLFWWindowHandle& window) {
    VkSurfaceKHR psurf;
    if (glfwCreateWindowSurface(
            device->get_physical_device()->get_instance()->get_instance(), *window, NULL, &psurf))
        throw std::runtime_error("Surface creation failed!");
    return vk::SurfaceKHR(psurf);
}

class GLFWSurface : public Surface {
  public:
    GLFWSurface(const DeviceHandle& device, const GLFWWindowHandle& window)
        : Surface(device, surface_from_glfw_window(device, window)), window(window) {
        SPDLOG_DEBUG("create surface ({})", fmt::ptr(this));
    }

  private:
    const GLFWWindowHandle window;
};

} // namespace merian
