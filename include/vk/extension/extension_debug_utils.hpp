#pragma once

#include <iostream>
#include <vk/context.hpp>
#include <vk/extension/extension.hpp>
#include <vulkan/vulkan.hpp>

using SEVERITY = vk::DebugUtilsMessageSeverityFlagBitsEXT;
using MESSAGE = vk::DebugUtilsMessageTypeFlagBitsEXT;

class ExtensionDebugUtils : public Extension {
  public:
    ~ExtensionDebugUtils() {
    }
    std::string name() const override {
        return "ExtensionDebugUtils";
    }
    std::vector<const char*> required_extension_names() const {
        return {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
            VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
        };
    }
    std::vector<const char*> required_layer_names() const {
        return {
            "VK_LAYER_KHRONOS_validation",
        };
    }
    void on_instance_created(vk::Instance&);
    void on_destroy(vk::Instance&);

  private:
    vk::DebugUtilsMessengerEXT messenger;
};
