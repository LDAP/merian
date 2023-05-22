#include <vk/extension/extension_debug_utils.hpp>

#include <iostream>
#include <vulkan/vulkan.hpp>

/*
    This is the function in which errors will go through to be displayed.
*/
VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                           VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                                                           VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                                                           void* /*pUserData*/) {
    // if (enableValidationLayers) {
    //     if (pCallbackData->messageIdNumber == 648835635) {
    //         // UNASSIGNED-khronos-Validation-debug-build-warning-message
    //         return VK_FALSE;
    //     }
    //     if (pCallbackData->messageIdNumber == 767975156) {
    //         // UNASSIGNED-BestPractices-vkCreateInstance-specialuse-extension
    //         return VK_FALSE;
    //     }
    // }

    std::cerr << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity)) << ": "
              << vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes)) << ":\n";
    std::cerr << "\t"
              << "messageIDName   = <" << pCallbackData->pMessageIdName << ">\n";
    std::cerr << "\t"
              << "messageIdNumber = " << pCallbackData->messageIdNumber << "\n";
    std::cerr << "\t"
              << "message         = <" << pCallbackData->pMessage << ">\n";
    if (0 < pCallbackData->queueLabelCount) {
        std::cerr << "\t"
                  << "Queue Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->queueLabelCount; i++) {
            std::cerr << "\t\t"
                      << "labelName = <" << pCallbackData->pQueueLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->cmdBufLabelCount) {
        std::cerr << "\t"
                  << "CommandBuffer Labels:\n";
        for (uint8_t i = 0; i < pCallbackData->cmdBufLabelCount; i++) {
            std::cerr << "\t\t"
                      << "labelName = <" << pCallbackData->pCmdBufLabels[i].pLabelName << ">\n";
        }
    }
    if (0 < pCallbackData->objectCount) {
        std::cerr << "\t"
                  << "Objects:\n";
        for (uint8_t i = 0; i < pCallbackData->objectCount; i++) {
            std::cerr << "\t\t"
                      << "Object " << i << "\n";
            std::cerr << "\t\t\t"
                      << "objectType   = "
                      << vk::to_string(static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType)) << "\n";
            std::cerr << "\t\t\t"
                      << "objectHandle = " << pCallbackData->pObjects[i].objectHandle << "\n";
            if (pCallbackData->pObjects[i].pObjectName) {
                std::cerr << "\t\t\t"
                          << "objectName   = <" << pCallbackData->pObjects[i].pObjectName << ">\n";
            }
        }
    }
    return VK_TRUE;
}

void ExtensionDebugUtils::on_instance_created(vk::Instance& instance) {
    vk::DebugUtilsMessengerCreateInfoEXT create_info{
        {},
        SEVERITY::eWarning | SEVERITY::eError,
        MESSAGE::eGeneral | MESSAGE::ePerformance | MESSAGE::eValidation,
        &debugUtilsMessengerCallback,
    };
    messenger = instance.createDebugUtilsMessengerEXT(create_info);
}
