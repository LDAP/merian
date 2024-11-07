#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkRobustnessAccess2 : public Extension {
  public:
    ExtensionVkRobustnessAccess2() : Extension("ExtensionVkRobustnessAccess2") {}
    ~ExtensionVkRobustnessAccess2() {}
    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override {
        return {VK_EXT_ROBUSTNESS_2_EXTENSION_NAME};
    }

    void* pnext_get_features_2(void* const p_next) override {
        supported_image_robustness_features.setPNext(p_next);
        return &supported_image_robustness_features;
    }

    void* pnext_device_create_info(void* const p_next) override {
        if (supported_image_robustness_features.robustImageAccess2 == VK_TRUE) {
            SPDLOG_DEBUG("robustImageAccess2 supported. Enabling feature");
            enable_image_robustness_features.robustImageAccess2 = VK_TRUE;
        } else {
            SPDLOG_WARN("robustImageAccess2 requested but not supported");
        }

        if (supported_image_robustness_features.robustBufferAccess2 == VK_TRUE) {
            SPDLOG_DEBUG("robustBufferAccess2 supported. Enabling feature");
            enable_image_robustness_features.robustBufferAccess2 = VK_TRUE;
        } else {
            SPDLOG_WARN("robustBufferAccess2 requested but not supported");
        }

        if (supported_image_robustness_features.nullDescriptor == VK_TRUE) {
            SPDLOG_DEBUG("nullDescriptor supported. Enabling feature");
            enable_image_robustness_features.nullDescriptor = VK_TRUE;
        } else {
            SPDLOG_WARN("nullDescriptor requested but not supported");
        }

        enable_image_robustness_features.pNext = p_next;
        return &enable_image_robustness_features;
    }

  private:
    vk::PhysicalDeviceRobustness2FeaturesEXT supported_image_robustness_features;
    vk::PhysicalDeviceRobustness2FeaturesEXT enable_image_robustness_features;
};

} // namespace merian
