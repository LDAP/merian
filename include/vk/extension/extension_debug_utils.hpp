#pragma once

#include <iostream>
#include <vk/context.hpp>
#include <vk/extension/extension.hpp>
#include <vulkan/vulkan.hpp>

using SEVERITY = vk::DebugUtilsMessageSeverityFlagBitsEXT;
using MESSAGE = vk::DebugUtilsMessageTypeFlagBitsEXT;

class ExtensionDebugUtils : public Extension {
  public:
    ExtensionDebugUtils() {
        create_info = {
            {},
            SEVERITY::eWarning | SEVERITY::eError,
            MESSAGE::eGeneral | MESSAGE::ePerformance | MESSAGE::eValidation,
            &ExtensionDebugUtils::messenger_callback,
        };
    }

    // Overrides
    ~ExtensionDebugUtils() {}
    std::string name() const override {
        return "ExtensionDebugUtils";
    }
    std::vector<const char*> required_instance_extension_names() const override {
        return {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
            VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
        };
    }
    std::vector<const char*> required_layer_names() const override {
        return {
            "VK_LAYER_KHRONOS_validation",
        };
    }
    std::vector<const char*> required_device_extension_names() const override {
        return {};
    }
    void on_instance_created(vk::Instance&) override;
    void on_destroy(vk::Instance&) override;
    void* on_create_instance(void* p_next) override;

    // Own methods
    static VKAPI_ATTR VkBool32 VKAPI_CALL messenger_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                             VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                             VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                                                             void* /*pUserData*/);

  private:
    vk::DebugUtilsMessengerCreateInfoEXT create_info;
    vk::DebugUtilsMessengerEXT messenger;
};
