#include "merian/vk/window/glfw_window.hpp"
#include "merian/vk/window/glfw_surface.hpp"

namespace merian {

GLFWWindow::operator GLFWwindow*() const {
    return window;
}

GLFWwindow* GLFWWindow::get_window() const {
    return window;
}

SurfaceHandle GLFWWindow::get_surface() {
    return std::make_shared<GLFWSurface>(context,
                                         std::static_pointer_cast<GLFWWindow>(shared_from_this()));
}

vk::Extent2D GLFWWindow::framebuffer_extent() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    return vk::Extent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

} // namespace merian
