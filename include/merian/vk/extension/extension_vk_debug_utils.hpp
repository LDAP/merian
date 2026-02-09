#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension.hpp"

#include <unordered_set>
#include <vulkan/vulkan.hpp>

namespace merian {

using SEVERITY = vk::DebugUtilsMessageSeverityFlagBitsEXT;
using MESSAGE = vk::DebugUtilsMessageTypeFlagBitsEXT;

class ExtensionVkDebugUtils : public ContextExtension {
  public:
    // Set assert_message to true to throw if an message with severity error is emitted.
    ExtensionVkDebugUtils(bool assert_message = true,
                          const std::unordered_set<int32_t>& ignore_message_ids = {648835635,
                                                                                   767975156})
        : ContextExtension("ExtensionVkDebugUtils"), user_data(ignore_message_ids, assert_message) {
        create_info = {
            {},
            SEVERITY::eError | SEVERITY::eWarning | SEVERITY::eInfo | SEVERITY::eVerbose,
            MESSAGE::eGeneral | MESSAGE::ePerformance | MESSAGE::eValidation,
            &ExtensionVkDebugUtils::messenger_callback,
            &this->user_data,
        };
    }

    // Overrides
    ~ExtensionVkDebugUtils() {
        (**instance).destroyDebugUtilsMessengerEXT(messenger);
    }

    InstanceSupportInfo
    query_instance_support(const InstanceSupportQueryInfo& query_info) override {
        InstanceSupportInfo info;
        info.required_extensions = {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
            VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
        };
        info.required_layers = {
            "VK_LAYER_KHRONOS_validation",
        };
        info.supported =
            query_info.supported_extensions.contains(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        return info;
    }

    void on_instance_created(const InstanceHandle& /*unused*/) override;

    void* pnext_instance_create_info(void* const p_next) override;

    // Own methods
    template <typename T>
    void set_object_name(const vk::Device& device, T handle, std::string name) {
        vk::DebugUtilsObjectNameInfoEXT info_ext(
            handle.objectType, uint64_t(static_cast<typename T::CType>(handle)), name.c_str());
        device.setDebugUtilsObjectNameEXT(info_ext);
    }

    void cmd_begin_label(const vk::CommandBuffer& cmd, const std::string& name) {
        const vk::DebugUtilsLabelEXT label{name.c_str()};
        cmd.beginDebugUtilsLabelEXT(label);
    }

    void cmd_end_label(const vk::CommandBuffer& cmd) {
        cmd.endDebugUtilsLabelEXT();
    }

  private:
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL
    messenger_callback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                       vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
                       vk::DebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                       void* pUserData);

  private:
    struct UserData {
        std::unordered_set<int32_t> ignore_message_ids;
        bool assert_message;
    };

    UserData user_data;
    InstanceHandle instance;

    vk::DebugUtilsMessengerCreateInfoEXT create_info;
    vk::DebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;

    std::vector<vk::ValidationFeatureEnableEXT> validation_feature_enables = {
        vk::ValidationFeatureEnableEXT::eDebugPrintf,
    };
    vk::ValidationFeaturesEXT validation_features;
};

} // namespace merian
