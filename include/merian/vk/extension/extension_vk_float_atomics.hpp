#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkFloatAtomics : public Extension {
  public:
    ExtensionVkFloatAtomics() : Extension("ExtensionVkFloatAtomics") {}
    ~ExtensionVkFloatAtomics() {}
    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override {
        return {
            VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        };
    }

    void* pnext_get_features_2(void* const p_next) override {
        supported_atomic_features.setPNext(p_next);
        return &supported_atomic_features;
    }

    void* pnext_device_create_info(void* const p_next) override {
        if (supported_atomic_features.shaderImageFloat32Atomics == VK_TRUE) {
            SPDLOG_DEBUG("shaderImageFloat32Atomics supported. Enabling feature");
            enable_atomic_features.shaderImageFloat32Atomics = VK_TRUE;
        } else {
            SPDLOG_WARN("shaderImageFloat32Atomics not supported");
        }

        if (supported_atomic_features.shaderImageFloat32AtomicAdd == VK_TRUE) {
            SPDLOG_DEBUG("shaderImageFloat32AtomicAdd supported. Enabling feature");
            enable_atomic_features.shaderImageFloat32AtomicAdd = VK_TRUE;
        } else {
            SPDLOG_WARN("shaderImageFloat32AtomicAdd not supported");
        }

        enable_atomic_features.pNext = p_next;
        return &enable_atomic_features;
    }

  private:
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT supported_atomic_features;
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT enable_atomic_features;
};

} // namespace merian
