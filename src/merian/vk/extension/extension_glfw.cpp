#include "merian/vk/extension/extension_glfw.hpp"
#include "merian/vk/window/glfw_window.hpp"

#include <spdlog/spdlog.h>

namespace merian {

ExtensionGLFW::ExtensionGLFW() : ContextExtension() {
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

void ExtensionGLFW::on_context_initializing(const vk::detail::DispatchLoaderDynamic& loader,
                                            [[maybe_unused]] const FileLoaderHandle& file_loader,
                                            [[maybe_unused]] const ContextCreateInfo& create_info) {

    SPDLOG_DEBUG("Querying Vulkan support");
    glfwInitVulkanLoader(loader.vkGetInstanceProcAddr);
    glfw_vulkan_support = glfwVulkanSupported();
    if (glfw_vulkan_support == GLFW_FALSE) {
        SPDLOG_WARN("...failed! GLFW reports to have no Vulkan support!");
    } else {
        SPDLOG_DEBUG("...success!");
    }
}

InstanceSupportInfo
ExtensionGLFW::query_instance_support(const InstanceSupportQueryInfo& /*query_info*/) {
    InstanceSupportInfo info;
    info.supported = (glfw_vulkan_support == GLFW_TRUE);

    if (glfw_vulkan_support == GLFW_TRUE) {
        uint32_t count;
        const char** extensions = glfwGetRequiredInstanceExtensions(&count);
        info.required_extensions.insert(info.required_extensions.end(), extensions,
                                        extensions + count);
    }

    return info;
}

DeviceSupportInfo ExtensionGLFW::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo info;

    if (glfw_vulkan_support == GLFW_FALSE) {
        info.supported = false;
        return info;
    }

    info.required_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    // Check presentation support
    info.supported =
        glfwGetPhysicalDevicePresentationSupport(
            **(query_info.physical_device->get_instance()), **query_info.physical_device,
            query_info.queue_info.queue_family_idx_GCT) == GLFW_TRUE;

    return info;
}

bool ExtensionGLFW::accept_graphics_queue(const InstanceHandle& instance,
                                          const PhysicalDeviceHandle& physical_device,
                                          std::size_t queue_family_index) {
    return glfw_vulkan_support == GLFW_FALSE ||
           glfwGetPhysicalDevicePresentationSupport(**instance, **physical_device,
                                                    queue_family_index) == GLFW_TRUE;
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
