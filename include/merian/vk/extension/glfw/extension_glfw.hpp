#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/glfw/glfw_window.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

namespace merian {

/*
 * @brief      Initializes GLFW and makes sure the graphics queue supports present.
 */
class ExtensionGLFW : public ContextExtension {
  public:
    struct glfw_error : public std::runtime_error {
        glfw_error(int id, const char* desc)
            : std::runtime_error(fmt::format("GLFW: {}: {}", id, desc)), desc(desc), id(id) {}

        const char* desc;
        const int id;
    };

  private:
    static void glfw_error_callback(int id, const char* desc) {
        throw glfw_error(id, desc);
    }

  public:
    ExtensionGLFW();

    ~ExtensionGLFW();

    void on_context_initializing(const PFN_vkGetInstanceProcAddr loader,
                                 const FileLoaderHandle& file_loader,
                                 const ContextCreateInfo& create_info) override;

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    bool accept_graphics_queue(const InstanceHandle& instance,
                               const PhysicalDeviceHandle& physical_device,
                               std::size_t queue_family_indext) override;

    // ----------------------------------------

    GLFWWindowHandle create_window(const DeviceHandle& device) const;

  private:
    int glfw_initialized = GLFW_FALSE;
    int glfw_vulkan_support = GLFW_FALSE;
};

} // namespace merian
