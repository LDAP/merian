#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkFilterCubic : public Extension {
  public:
    ExtensionVkFilterCubic() : Extension("ExtensionVkFilterCubic") {}
    ~ExtensionVkFilterCubic() {}
    std::vector<const char*>
    required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const override {
        return {VK_EXT_FILTER_CUBIC_EXTENSION_NAME};
    }
};

} // namespace merian
