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
  private:
    static void glfw_error_callback(int id, const char* desc) {
        std::string error = fmt::format("GLFW: {}: {}", id, desc);
        SPDLOG_ERROR(error);
        throw std::runtime_error(error);
    }

  public:
    ExtensionVkGLFW();

    ~ExtensionVkGLFW();

    std::vector<const char*> required_instance_extension_names() const override;

    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override;

    bool accept_graphics_queue(const vk::Instance& instance,
                               const vk::PhysicalDevice& physical_device,
                               std::size_t queue_family_indext) override;
};

} // namespace merian
