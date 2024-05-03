#include "merian/vk/extension/extension_vk_glfw.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/window/glfw_surface.hpp"

#include <spdlog/spdlog.h>

namespace merian {

void ExtensionVkGLFW::on_instance_created(const vk::Instance& instance) {
    auto psurf = VkSurfaceKHR(surface);
    if (glfwCreateWindowSurface(instance, window, NULL, &psurf))
        throw std::runtime_error("Surface creation failed!");
    surface = vk::SurfaceKHR(psurf);
    SPDLOG_DEBUG("created surface");
}

bool ExtensionVkGLFW::accept_graphics_queue(const vk::PhysicalDevice& physical_device,
                                            std::size_t queue_family_index) {
    return physical_device.getSurfaceSupportKHR(queue_family_index, surface);
}

void ExtensionVkGLFW::on_destroy_instance(const vk::Instance& instance) {
    if (window) {
        instance.destroySurfaceKHR(surface);
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

std::tuple<GLFWWindowHandle, SurfaceHandle> ExtensionVkGLFW::get() {
    if (!window) {
        std::runtime_error{"ExtensionVkGLFW:get() can only be called exactly once!"};
    }

    GLFWwindow* window = this->window;
    vk::SurfaceKHR surface = this->surface;

    this->window = nullptr;
    this->surface = vk::SurfaceKHR();

    assert(!weak_context.expired());
    SharedContext context = weak_context.lock();

    std::shared_ptr<GLFWWindow> shared_window = std::make_shared<GLFWWindow>(context, window);
    std::shared_ptr<Surface> shared_surface = std::shared_ptr<Surface>(new GLFWSurface(context, surface, shared_window));

    return {shared_window, shared_surface};
}

} // namespace merian
