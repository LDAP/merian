#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkPushDescriptor : public Extension {
  public:
    ExtensionVkPushDescriptor() : Extension("ExtensionVkPushDescriptor") {}
    ~ExtensionVkPushDescriptor() {}

    std::vector<const char*>
    required_device_extension_names(vk::PhysicalDevice) const override {
        return {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    }
};

} // namespace merian
