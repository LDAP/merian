#pragma once

#include "vk/extension/extension.hpp"
class ExtensionFloatAtomics : public Extension {
  public:
    ExtensionFloatAtomics() {
        atomic_features.shaderImageFloat32Atomics = VK_TRUE;
        atomic_features.shaderImageFloat32AtomicAdd = VK_TRUE;
    }
    ~ExtensionFloatAtomics() {}
    std::string name() const override {
        return "ExtensionFloatAtomics";
    }
    std::vector<const char*> required_device_extension_names() const override {
        return {
            VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        };
    }
    void* on_create_device(void* p_next) override {
        atomic_features.pNext = p_next;
        return &atomic_features;
    }
    bool extension_supported(vk::PhysicalDevice& physical_device) override {
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_features;
        vk::PhysicalDeviceFeatures2 physical_device_features_2;
        physical_device_features_2.setPNext(&atomic_features);
        physical_device.getFeatures2(&physical_device_features_2);

        return atomic_features.shaderImageFloat32AtomicAdd == VK_TRUE;
    }

  private:
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_features;
};
