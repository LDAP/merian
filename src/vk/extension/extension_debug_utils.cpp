#include <spdlog/spdlog.h>
#include <vk/extension/extension_debug_utils.hpp>

#include <iostream>
#include <vulkan/vulkan.hpp>

/*
    This is the function in which errors will go through to be displayed.
*/
VKAPI_ATTR VkBool32 VKAPI_CALL ExtensionDebugUtils::messenger_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData, void* /*pUserData*/) {

    if (pCallbackData->messageIdNumber == 648835635) {
        // UNASSIGNED-khronos-Validation-debug-build-warning-message
        return VK_FALSE;
    }
    if (pCallbackData->messageIdNumber == 767975156) {
        // UNASSIGNED-BestPractices-vkCreateInstance-specialuse-extension
        return VK_FALSE;
    }

    spdlog::level::level_enum severity;
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        severity = spdlog::level::level_enum::err;
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        severity = spdlog::level::level_enum::warn;
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        severity = spdlog::level::level_enum::info;
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        severity = spdlog::level::level_enum::trace;
    } else {
        severity = spdlog::level::level_enum::err;
    }

    std::string msg_type = vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes));
    spdlog::log(severity, "[{}] [{}] [{}]\n{}", msg_type, pCallbackData->pMessageIdName, pCallbackData->messageIdNumber,
                pCallbackData->pMessage);

    if (0 < pCallbackData->queueLabelCount) {
        std::string additional_info;
        additional_info += "Queue Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->queueLabelCount; i++) {
            additional_info += "\t\t";
            additional_info += "labelName = <";
            additional_info += pCallbackData->pQueueLabels[i].pLabelName ;
            additional_info += ">\n";
        }
        spdlog::log(severity, additional_info);
    }
    if (0 < pCallbackData->cmdBufLabelCount) {
        std::string additional_info;
        additional_info += "CommandBuffer Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->cmdBufLabelCount; i++) {
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
        for (uint8_t i = 0; i < pCallbackData->objectCount; i++) {
            additional_info += "\t";
            additional_info += "Object ";
            additional_info += i;
            additional_info += "\n";
            additional_info += "\t\t";
            additional_info += "objectType   = ";
            additional_info += vk::to_string(static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType));
            additional_info += "\n";
            additional_info += "\t\t";
            additional_info += "objectHandle = ";
            additional_info += pCallbackData->pObjects[i].objectHandle;
            additional_info += "\n";
            if (pCallbackData->pObjects[i].pObjectName) {
                additional_info += "\t\t";
                additional_info += "objectName   = <";
                additional_info += pCallbackData->pObjects[i].pObjectName;
                additional_info += ">\n";
            }
        }
        spdlog::log(severity, additional_info);
    }

    return VK_TRUE;
}

void* ExtensionDebugUtils::on_create_instance(void* p_next) {
    this->create_info.setPNext(p_next);
    return &this->create_info;
}

void ExtensionDebugUtils::on_instance_created(vk::Instance& instance) {
    this->create_info.setPNext(nullptr);
    messenger = instance.createDebugUtilsMessengerEXT(create_info);
}

void ExtensionDebugUtils::on_destroy(vk::Instance& instance) {
    instance.destroyDebugUtilsMessengerEXT(messenger);
}
