#pragma once

#include "merian/vk/extension/extension.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

namespace merian {

/*
 * @brief      Initializes GLFW and makes sure the graphics queue supports present.
 */
class ExtensionVkGLFW : public Extension {
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
    ExtensionVkGLFW();

    ~ExtensionVkGLFW();

    std::vector<const char*> required_instance_extension_names() const override;

    std::vector<const char*>
    required_device_extension_names(const vk::PhysicalDevice&) const override;

    bool accept_graphics_queue(const vk::Instance& instance,
                               const vk::PhysicalDevice& physical_device,
                               std::size_t queue_family_indext) override;
};

} // namespace merian
