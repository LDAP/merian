#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkFloatAtomics : public Extension {
  public:
    ExtensionVkFloatAtomics() : Extension("ExtensionVkFloatAtomics") {
        atomic_features.shaderImageFloat32Atomics = VK_TRUE;
        atomic_features.shaderImageFloat32AtomicAdd = VK_TRUE;
    }
    ~ExtensionVkFloatAtomics() {}
    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override {
        return {
            VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        };
    }
    void* pnext_device_create_info(void* const p_next) override {
        atomic_features.pNext = p_next;
        return &atomic_features;
    }
    bool extension_supported(const vk::PhysicalDevice& physical_device) override {
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_features;
        vk::PhysicalDeviceFeatures2 physical_device_features_2;
        physical_device_features_2.setPNext(&atomic_features);
        physical_device.getFeatures2(&physical_device_features_2);

        return atomic_features.shaderImageFloat32AtomicAdd == VK_TRUE;
    }

  private:
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_features;
};

} // namespace merian
