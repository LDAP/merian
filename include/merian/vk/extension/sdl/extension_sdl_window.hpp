#pragma once

#include "merian/vk/extension/extension.hpp"

#include <memory>

namespace merian {

class SDLWindow;
using SDLWindowHandle = std::shared_ptr<SDLWindow>;

class ExtensionSDLWindow : public ContextExtension {
  public:
    ExtensionSDLWindow();

    ~ExtensionSDLWindow();

    std::vector<std::string> request_extensions() override;

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    bool accept_graphics_queue(const InstanceHandle& instance,
                               const PhysicalDeviceHandle& physical_device,
                               std::size_t queue_family_index) override;

    SDLWindowHandle create_window(const DeviceHandle& device,
                                  int width = 1280,
                                  int height = 720,
                                  const char* title = "") const;

  private:
    bool video_initialized = false;
    bool sdl_vulkan_support = false;
};

} // namespace merian
