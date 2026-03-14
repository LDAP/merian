#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/sdl/extension_sdl.hpp"
#include "merian/vk/window/window_provider.hpp"

#include <memory>

namespace merian {

class SDLWindow;
using SDLWindowHandle = std::shared_ptr<SDLWindow>;

class ExtensionSDLWindow : public ContextExtension, public WindowProvider {
  public:
    static constexpr const char* name = "SDL Window";

    ExtensionSDLWindow();

    ~ExtensionSDLWindow();

    std::vector<std::string> request_extensions() override;

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    bool accept_graphics_queue(const InstanceHandle& instance,
                               const PhysicalDeviceHandle& physical_device,
                               std::size_t queue_family_index) override;

    WindowHandle create_window(const DeviceHandle& device,
                               const WindowCreateInfo& create_info = {}) const override;

  private:
    std::shared_ptr<ExtensionSDL> sdl_ext;
    bool video_initialized = false;
    bool sdl_vulkan_support = false;
};

} // namespace merian
