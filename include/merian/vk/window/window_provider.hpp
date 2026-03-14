#pragma once

#include "merian/fwd.hpp"
#include "merian/vk/window/window.hpp"

namespace merian {

/*
 * Mixin for ContextExtensions that can create windows.
 */
class WindowProvider {
  public:
    virtual ~WindowProvider() = default;

    virtual WindowHandle create_window(const DeviceHandle& device,
                                       const WindowCreateInfo& create_info = {}) const = 0;
};

} // namespace merian
