#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkShaderMaximalReconvergence : public Extension {
  public:
    ExtensionVkShaderMaximalReconvergence() : Extension("ExtensionVkShaderMaximalReconvergence") {}
    ~ExtensionVkShaderMaximalReconvergence() {}
    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override {
        return {VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME};
    }

    void* pnext_get_features_2(void* const p_next) override {
        supported_features.setPNext(p_next);
        return &supported_features;
    }

    void* pnext_device_create_info(void* const p_next) override {
        if (supported_features.shaderMaximalReconvergence) {
            SPDLOG_DEBUG("shaderMaximalReconvergence supported. Enabling feature");
            enabled_features.shaderMaximalReconvergence = true;
        } else {
            SPDLOG_ERROR("shaderMaximalReconvergence requested but not supported");
        }

        enabled_features.pNext = p_next;
        return &enabled_features;
    }

  private:
    vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR supported_features;
    vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR enabled_features;
};

} // namespace merian
