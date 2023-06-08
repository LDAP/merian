#include "merian/vk/extension/extension_vk_glfw.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/command/queue_container.hpp"

#include <spdlog/spdlog.h>

namespace merian {

void ExtensionVkGLFW::on_instance_created(const vk::Instance& instance) {
    auto psurf = VkSurfaceKHR(surface);
    if (glfwCreateWindowSurface(instance, window, NULL, &psurf))
        throw std::runtime_error("Surface creation failed!");
    surface = vk::SurfaceKHR(psurf);
    SPDLOG_DEBUG("created surface");
}

void ExtensionVkGLFW::on_physical_device_selected(const Context::PhysicalDeviceContainer& pd_container) {
    this->physical_device = pd_container.physical_device;
}

bool ExtensionVkGLFW::accept_graphics_queue(const vk::PhysicalDevice& physical_device,
                                            std::size_t queue_family_index) {
    if (physical_device.getSurfaceSupportKHR(queue_family_index, surface)) {
        return true;
    }
    return false;
}

void ExtensionVkGLFW::on_destroy_instance(const vk::Instance& instance) {
    if (window) {
        instance.destroySurfaceKHR(surface);
        glfwDestroyWindow(window);
    }
    glfwTerminate();
}

} // namespace merian
