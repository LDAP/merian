#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * @brief      This class describes an extension to access VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME.
 */
class ExtensionVkAccelerationStructure : public Extension {
  public:
    ExtensionVkAccelerationStructure() : Extension("ExtensionVkAccelerationStructure") {
        acceleration_structure_features.accelerationStructure = VK_TRUE;
    }
    ~ExtensionVkAccelerationStructure() {}

    std::vector<const char*> required_device_extension_names() const override {
        return {
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        };
    }

    // LIFECYCLE

    void on_physical_device_selected(const vk::PhysicalDevice& physical_device) override {
        vk::PhysicalDeviceProperties2KHR props2;
        props2.pNext = &acceleration_structure_properties;
        physical_device.getProperties2(&props2);
    }
    void* on_create_device(void* const p_next) override {
        acceleration_structure_features.pNext = p_next;
        return &acceleration_structure_features;
    }

  private:
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features;

  public:
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties;
};

} // namespace merian
