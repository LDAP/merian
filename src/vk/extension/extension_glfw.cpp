#include "vk/extension/extension_glfw.hpp"
#include <spdlog/spdlog.h>

void ExtensionGLFW::on_instance_created(vk::Instance& instance) {
    auto psurf = VkSurfaceKHR(surface);
    if (glfwCreateWindowSurface(instance, window, NULL, &psurf))
        throw std::runtime_error("Surface creation failed!");
    surface = vk::SurfaceKHR(psurf);
    spdlog::debug("created surface");
}

void ExtensionGLFW::on_destroy(vk::Instance& instance) {
    spdlog::debug("destroy surface");
    instance.destroySurfaceKHR(surface);
}

bool ExtensionGLFW::accept_graphics_queue(vk::PhysicalDevice& physical_device, std::size_t queue_family_index) {
    if (physical_device.getSurfaceSupportKHR(queue_family_index, surface)) {
        return true;
    }
    return false;
}
