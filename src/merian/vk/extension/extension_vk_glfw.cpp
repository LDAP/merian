#include "merian/vk/extension/extension_vk_glfw.hpp"

#include <spdlog/spdlog.h>

namespace merian {

ExtensionVkGLFW::ExtensionVkGLFW() : Extension("ExtensionVkGLFW") {
    glfwSetErrorCallback(glfw_error_callback);

    SPDLOG_DEBUG("Initialize GLFW");
    if (glfwInit() == GLFW_FALSE)
        throw std::runtime_error("GLFW initialization failed!");
    if (glfwVulkanSupported() == GLFW_FALSE)
        throw std::runtime_error("GLFW reports to have no Vulkan support! Maybe it couldn't "
                                 "find the Vulkan loader!");
}

ExtensionVkGLFW::~ExtensionVkGLFW() {
    SPDLOG_DEBUG("Terminate GLFW");
    glfwTerminate();
}

std::vector<const char*> ExtensionVkGLFW::required_instance_extension_names() const {
    std::vector<const char*> required_extensions;
    uint32_t count;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);
    required_extensions.insert(required_extensions.end(), extensions, extensions + count);
    return required_extensions;
}

std::vector<const char*>
ExtensionVkGLFW::required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const {
    return {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
}

bool ExtensionVkGLFW::accept_graphics_queue(const vk::Instance& instance,
                                            const vk::PhysicalDevice& physical_device,
                                            std::size_t queue_family_index) {
    return glfwGetPhysicalDevicePresentationSupport(instance, physical_device,
                                                    queue_family_index) == GLFW_TRUE;
}

} // namespace merian
