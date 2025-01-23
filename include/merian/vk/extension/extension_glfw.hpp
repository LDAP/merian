#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/window/glfw_window.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

namespace merian {

/*
 * @brief      Initializes GLFW and makes sure the graphics queue supports present.
 */
class ExtensionGLFW : public Extension {
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

    std::vector<const char*> required_instance_extension_names() const override;

    std::vector<const char*>
    required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const override;

    bool accept_graphics_queue(const vk::Instance& instance,
                               const PhysicalDevice& physical_device,
                               std::size_t queue_family_indext) override;

    bool extension_supported(const vk::Instance& instance,
                             const PhysicalDevice& physical_device,
                             const ExtensionContainer& extension_container,
                             const QueueInfo& queue_info) override;

    void on_context_created(const ContextHandle& context,
                            const ExtensionContainer& extension_container) override;

    // ----------------------------------------

    GLFWWindowHandle create_window() const;

  private:
    int glfw_initialized = GLFW_FALSE;
    int glfw_vulkan_support = GLFW_FALSE;

    WeakContextHandle weak_context;
};

} // namespace merian
