#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/window/window_provider.hpp"

namespace merian {

class ExtensionHostWindows : public ContextExtension, public WindowProvider {
  public:
    static constexpr const char* name = "merian-host-windows";

    std::vector<std::string> request_extensions() override;

    WindowHandle create_window(const DeviceHandle& device,
                               const WindowCreateInfo& create_info = {}) const override;
};

} // namespace merian
