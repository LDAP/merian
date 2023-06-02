#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * @brief      This class describes an extension to access
 * VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME.
 * 
 * Note that you must enable Ray Query or Ray Tracing Pipeline extension for this to work.
 */
class ExtensionVkAccelerationStructure : public Extension {
  public:
    ExtensionVkAccelerationStructure() : Extension("ExtensionVkAccelerationStructure") {
        acceleration_structure_features.accelerationStructure = VK_TRUE;
    }
    ~ExtensionVkAccelerationStructure() {}

    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override {
        return {
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        };
    }

    // LIFECYCLE

    void
    on_physical_device_selected(const Context::PhysicalDeviceContainer& pd_container) override {
        vk::PhysicalDeviceProperties2KHR props2;
        props2.pNext = &acceleration_structure_properties;
        pd_container.physical_device.getProperties2(&props2);
    }
    void* pnext_device_create_info(void* const p_next) override {
        acceleration_structure_features.pNext = p_next;
        return &acceleration_structure_features;
    }

  private:
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features;

  public:
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties;
};

} // namespace merian
