#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension.hpp"

#include <vulkan/vulkan.hpp>

#include <iostream>
#include <unordered_set>

namespace merian {

using SEVERITY = vk::DebugUtilsMessageSeverityFlagBitsEXT;
using MESSAGE = vk::DebugUtilsMessageTypeFlagBitsEXT;

class ExtensionVkDebugUtils : public Extension {
  public:
    ExtensionVkDebugUtils(std::unordered_set<int32_t> ignore_message_ids = {648835635, 767975156})
        : Extension("ExtensionVkDebugUtils"), ignore_message_ids(ignore_message_ids) {
        create_info = {
            {},
            SEVERITY::eWarning | SEVERITY::eError,
            MESSAGE::eGeneral | MESSAGE::ePerformance | MESSAGE::eValidation,
            &ExtensionVkDebugUtils::messenger_callback,
            &this->ignore_message_ids,
        };
    }

    // Overrides
    ~ExtensionVkDebugUtils() {}
    std::vector<const char*> required_instance_extension_names() const override {
        return {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
            VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
        };
    }
    std::vector<const char*> required_instance_layer_names() const override {
        return {
            "VK_LAYER_KHRONOS_validation",
        };
    }
    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override {
        return {};
    }
    void on_instance_created(const vk::Instance&) override;
    void on_destroy_instance(const vk::Instance&) override;
    void* pnext_instance_create_info(void* const p_next) override;

    // Own methods
    template <typename T> void set_object_name(vk::Device& device, T handle, std::string name) {
        vk::DebugUtilsObjectNameInfoEXT infoEXT(
            handle.objectType, uint64_t(static_cast<typename T::CType>(handle)), name.c_str());
        device.setDebugUtilsObjectNameEXT(infoEXT);
    }

  private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL
    messenger_callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                       VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                       VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                       void* pUserData);

  private:
    std::unordered_set<int32_t> ignore_message_ids;
    vk::DebugUtilsMessengerCreateInfoEXT create_info;
    vk::DebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
};

} // namespace merian
