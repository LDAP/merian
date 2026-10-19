#include "merian/vk/extension/extension_host_windows.hpp"

#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/window/host_window.hpp"

namespace merian {

std::vector<std::string> ExtensionHostWindows::request_extensions() {
    return {ExtensionResources::name};
}

WindowHandle ExtensionHostWindows::create_window(const DeviceHandle& /*device*/,
                                                 const WindowCreateInfo& create_info) const {
    return std::make_shared<HostWindow>(create_info);
}

} // namespace merian
