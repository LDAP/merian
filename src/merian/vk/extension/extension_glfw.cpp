#include "merian/vk/extension/extension_glfw.hpp"
#include "merian/vk/window/glfw_window.hpp"

#include <spdlog/spdlog.h>

namespace merian {

ExtensionGLFW::ExtensionGLFW() : ContextExtension("ExtensionGLFW") {
    glfwSetErrorCallback(glfw_error_callback);

    SPDLOG_DEBUG("Initialize GLFW");
    glfw_initialized = glfwInit();
    if (glfw_initialized == GLFW_FALSE) {
        SPDLOG_WARN("GLFW initialization failed!");
    }

    SPDLOG_DEBUG("Initialized GLFW: {}", glfw_initialized == GLFW_TRUE);
}

ExtensionGLFW::~ExtensionGLFW() {
    SPDLOG_DEBUG("Terminate GLFW");
    if (glfw_initialized == GLFW_TRUE)
        glfwTerminate();
}

void ExtensionGLFW::on_context_initializing(
    [[maybe_unused]] const ExtensionContainer& extension_container,
    const vk::detail::DispatchLoaderDynamic& loader) {

    SPDLOG_DEBUG("Querying Vulkan support");
    glfwInitVulkanLoader(loader.vkGetInstanceProcAddr);
    glfw_vulkan_support = glfwVulkanSupported();
    if (glfw_vulkan_support == GLFW_FALSE) {
        SPDLOG_WARN("...failed! GLFW reports to have no Vulkan support!");
    } else {
        SPDLOG_DEBUG("...success!");
    }
}

std::vector<const char*> ExtensionGLFW::enable_instance_extension_names(
    const std::unordered_set<std::string>& /*supported_instance_extensions*/) const {
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
ExtensionGLFW::enable_device_extension_names(const PhysicalDeviceHandle& /*unused*/) const {
    if (glfw_vulkan_support == GLFW_FALSE) {
        return {};
    }

    return {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
}

bool ExtensionGLFW::accept_graphics_queue(const InstanceHandle& instance,
                                          const PhysicalDeviceHandle& physical_device,
                                          std::size_t queue_family_index) {
    return glfw_vulkan_support == GLFW_FALSE ||
           glfwGetPhysicalDevicePresentationSupport(**instance, **physical_device,
                                                    queue_family_index) == GLFW_TRUE;
}

bool ExtensionGLFW::extension_supported(const PhysicalDeviceHandle& physical_device,
                                        [[maybe_unused]] const QueueInfo& queue_info) {
    if (ContextExtension::extension_supported(physical_device, queue_info)) {
        return glfw_vulkan_support == GLFW_TRUE &&
               glfwGetPhysicalDevicePresentationSupport(
                   **(physical_device->get_instance()), **physical_device,
                   queue_info.queue_family_idx_GCT) == GLFW_TRUE;
    }
    return false;
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
