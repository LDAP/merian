#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkNvComputeShaderDerivatives : public Extension {
  public:
    ExtensionVkNvComputeShaderDerivatives() : Extension("ExtensionVkNvComputeShaderDerivatives") {}
    ~ExtensionVkNvComputeShaderDerivatives() {}
    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice /*unused*/) const override {
        return {VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME};
    }

    void* pnext_get_features_2(void* const p_next) override {
        supported_features.setPNext(p_next);
        return &supported_features;
    }

    bool extension_supported(const Context::PhysicalDeviceContainer& /*unused*/) override {
        return supported_features.computeDerivativeGroupLinear == VK_TRUE ||
               supported_features.computeDerivativeGroupQuads == VK_TRUE;
    }

    void* pnext_device_create_info(void* const p_next) override {
        if (supported_features.computeDerivativeGroupQuads == VK_TRUE) {
            SPDLOG_DEBUG("computeDerivativeGroupQuads supported. Enabling feature");
            enabled_features.computeDerivativeGroupQuads = VK_TRUE;
        } else {
            SPDLOG_ERROR("computeDerivativeGroupQuads requested but not supported");
        }
        if (supported_features.computeDerivativeGroupLinear == VK_TRUE) {
            SPDLOG_DEBUG("computeDerivativeGroupLinear supported. Enabling feature");
            enabled_features.computeDerivativeGroupLinear = VK_TRUE;
        } else {
            SPDLOG_ERROR("computeDerivativeGroupLinear requested but not supported");
        }

        enabled_features.pNext = p_next;
        return &enabled_features;
    }

  private:
    vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV supported_features;
    vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV enabled_features;
};

} // namespace merian
