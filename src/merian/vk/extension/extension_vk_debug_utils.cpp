#include "merian/vk/extension/extension_vk_debug_utils.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan.hpp>

namespace {
spdlog::level::level_enum get_severity(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity) {
    if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
        return spdlog::level::level_enum::err;
    }
    if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
        return spdlog::level::level_enum::warn;
    }
    if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo) {
        return spdlog::level::level_enum::info;
    }
    if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) {
        return spdlog::level::level_enum::trace;
    }
    return spdlog::level::level_enum::err;
}
} // namespace

namespace merian {

/*
    This is the function in which errors will go through to be displayed.
*/
VKAPI_ATTR vk::Bool32 VKAPI_CALL ExtensionVkDebugUtils::messenger_callback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
    vk::DebugUtilsMessengerCallbackDataEXT const* pCallbackData,
    void* pUserData) {

    UserData* user_data = static_cast<UserData*>(pUserData);
    if (user_data->ignore_message_ids.contains(pCallbackData->messageIdNumber)) {
        return VK_FALSE;
    }

    spdlog::level::level_enum severity = get_severity(messageSeverity);
    std::string msg_type =
        vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes));
    spdlog::log(severity, "[{}] [{}] [{}]\n{}", msg_type, pCallbackData->pMessageIdName,
                pCallbackData->messageIdNumber, pCallbackData->pMessage);

    if (0 < pCallbackData->queueLabelCount) {
        std::string additional_info;
        additional_info += "Queue Labels:\n";
        for (uint32_t i = 0; i < pCallbackData->queueLabelCount; i++) {
            additional_info += "\t\t";
            additional_info += "labelName = <";
            additional_info += pCallbackData->pQueueLabels[i].pLabelName;
            additional_info += ">\n";
        }
        spdlog::log(severity, additional_info);
    }
    if (0 < pCallbackData->cmdBufLabelCount) {
        std::string additional_info;
        additional_info += "CommandBuffer Labels:\n";
        for (uint32_t i = 0; i < pCallbackData->cmdBufLabelCount; i++) {
            additional_info += "\t\t";
            additional_info += "labelName = <";
            additional_info += pCallbackData->pCmdBufLabels[i].pLabelName;
            additional_info += ">\n";
        }
        spdlog::log(severity, additional_info);
    }
    if (0 < pCallbackData->objectCount) {
        std::string additional_info;
        additional_info += "Objects:\n";
        for (uint32_t i = 0; i < pCallbackData->objectCount; i++) {
            additional_info += "\t";
            additional_info += "Object ";
            additional_info += std::to_string(i);
            additional_info += "\n";
            additional_info += "\t\t";
            additional_info += "objectType   = ";
            additional_info +=
                vk::to_string(static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType));
            additional_info += "\n";
            additional_info += "\t\t";
            additional_info += "objectHandle = ";
            additional_info += fmt::format("{:x}", pCallbackData->pObjects[i].objectHandle);
            additional_info += "\n";
            if (pCallbackData->pObjects[i].pObjectName != nullptr) {
                additional_info += "\t\t";
                additional_info += "objectName   = ";
                additional_info += pCallbackData->pObjects[i].pObjectName;
                additional_info += "\n";
            }
        }
        spdlog::log(severity, additional_info);
    }

    assert(!user_data->assert_message ||
           !(messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError));

    return VK_FALSE;
}

void* ExtensionVkDebugUtils::pnext_instance_create_info(void* p_next) {
    validation_features.setEnabledValidationFeatures(validation_feature_enables);

    create_info.setPNext(p_next);
    validation_features.setPNext(&create_info);

    return &validation_features;
}

void ExtensionVkDebugUtils::on_instance_created(const InstanceHandle& instance) {
    this->create_info.setPNext(nullptr);
    messenger = (**instance).createDebugUtilsMessengerEXT(create_info);
    assert(this->instance == nullptr);

    this->instance = instance;
}

} // namespace merian
