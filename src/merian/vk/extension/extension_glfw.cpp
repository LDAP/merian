#include "merian/vk/extension/extension_glfw.hpp"
#include "merian/vk/window/glfw_window.hpp"

#include <spdlog/spdlog.h>

namespace merian {

ExtensionGLFW::ExtensionGLFW() : Extension("ExtensionGLFW") {
    glfwSetErrorCallback(glfw_error_callback);

    SPDLOG_DEBUG("Initialize GLFW");
    glfw_initialized = glfwInit();
    if (glfw_initialized == GLFW_FALSE) {
        SPDLOG_WARN("GLFW initialization failed!");
    } else {
        glfw_vulkan_support = glfwVulkanSupported();
        if (glfw_vulkan_support == GLFW_FALSE)
            SPDLOG_WARN("GLFW reports to have no Vulkan support! Maybe it couldn't "
                        "find the Vulkan loader!");
    }
}

ExtensionGLFW::~ExtensionGLFW() {
    SPDLOG_DEBUG("Terminate GLFW");
    if (glfw_initialized == GLFW_TRUE)
        glfwTerminate();
}

std::vector<const char*> ExtensionGLFW::required_instance_extension_names() const {
    if (glfw_vulkan_support == GLFW_FALSE) {
        return {};
    }

    std::vector<const char*> required_extensions;
    uint32_t count;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);
    required_extensions.insert(required_extensions.end(), extensions, extensions + count);
    return required_extensions;
}

std::vector<const char*>
ExtensionGLFW::required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const {
    if (glfw_vulkan_support == GLFW_FALSE) {
        return {};
    }

    return {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
}

bool ExtensionGLFW::accept_graphics_queue(const vk::Instance& instance,
                                          const PhysicalDevice& physical_device,
                                          std::size_t queue_family_index) {
    return glfw_vulkan_support == GLFW_FALSE ||
           glfwGetPhysicalDevicePresentationSupport(instance, *physical_device,
                                                    queue_family_index) == GLFW_TRUE;
}

bool ExtensionGLFW::extension_supported(
    const vk::Instance& instance,
    [[maybe_unused]] const PhysicalDevice& physical_device,
    [[maybe_unused]] const ExtensionContainer& extension_container,
    const QueueInfo& queue_info) {
    return glfw_vulkan_support == GLFW_TRUE &&
           glfwGetPhysicalDevicePresentationSupport(instance, *physical_device,
                                                    queue_info.queue_family_idx_GCT) == GLFW_TRUE;
}

void ExtensionGLFW::on_context_created(const ContextHandle& context,
                                       const ExtensionContainer& /*extension_container*/) {
    weak_context = context;
}

GLFWWindowHandle ExtensionGLFW::create_window() const {
    assert(!weak_context.expired());

    return std::shared_ptr<GLFWWindow>(new GLFWWindow(weak_context.lock()));
}

} // namespace merian
