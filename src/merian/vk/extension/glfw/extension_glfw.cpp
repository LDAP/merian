#include "merian/vk/extension/glfw/extension_glfw.hpp"
#include "merian/vk/extension/glfw/glfw_window.hpp"

#include <spdlog/spdlog.h>

namespace merian {

ExtensionGLFW::ExtensionGLFW() : ContextExtension() {
    glfwSetErrorCallback(glfw_error_callback);
}

ExtensionGLFW::~ExtensionGLFW() {
    SPDLOG_DEBUG("Terminate GLFW");
    if (glfw_initialized == GLFW_TRUE)
        glfwTerminate();
}

void ExtensionGLFW::on_context_initializing(const PFN_vkGetInstanceProcAddr loader,
                                            [[maybe_unused]] const FileLoaderHandle& file_loader,
                                            [[maybe_unused]] const ContextCreateInfo& create_info) {

    glfwInitVulkanLoader(loader);

    SPDLOG_DEBUG("Initialize GLFW {}...", glfwGetVersionString());
    glfw_initialized = glfwInit();

    if (glfw_initialized == GLFW_TRUE) {
        SPDLOG_DEBUG("Querying Vulkan support...");
        glfw_vulkan_support = glfwVulkanSupported();
    }

    if (glfw_initialized == GLFW_FALSE || glfw_vulkan_support == GLFW_FALSE) {
        SPDLOG_WARN("...failed! GLFW initialized: {}, Vulkan support: {}", bool(glfw_initialized),
                    bool(glfw_vulkan_support));
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
    if (glfw_vulkan_support == GLFW_FALSE) {
        return DeviceSupportInfo{false, "GLFW has no Vulkan support"};
    }

    DeviceSupportInfo info =
        DeviceSupportInfo::check(query_info, {}, {}, {VK_KHR_SWAPCHAIN_EXTENSION_NAME});
    if (!info.supported) {
        return info;
    }

    const std::optional<QueueInfo> queue = query_info.queues.get(vk::QueueFlagBits::eGraphics);
    if (!queue || glfwGetPhysicalDevicePresentationSupport(
                      **query_info.physical_device->get_instance(), **query_info.physical_device,
                      queue->family_index) != GLFW_TRUE) {
        return DeviceSupportInfo{false, "no graphics queue family can present"};
    }

    return info;
}

std::vector<QueueRequest>
ExtensionGLFW::request_queues([[maybe_unused]] const PhysicalDeviceHandle& physical_device) {
    if (glfw_vulkan_support == GLFW_FALSE) {
        return {};
    }
    return {{
        .capabilities = vk::QueueFlagBits::eGraphics,
        .prefer_family =
            [](const PhysicalDeviceHandle& device, const uint32_t family_index) {
                return glfwGetPhysicalDevicePresentationSupport(**device->get_instance(), **device,
                                                                family_index) == GLFW_TRUE;
            },
    }};
}

WindowHandle ExtensionGLFW::create_window(const DeviceHandle& device,
                                          const WindowCreateInfo& create_info) const {
    return std::shared_ptr<GLFWWindow>(new GLFWWindow(device, create_info));
}

} // namespace merian
