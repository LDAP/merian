#include "merian/vk/physical_device.hpp"
#include "spdlog/spdlog.h"

namespace merian {
PhysicalDevice::PhysicalDevice(const InstanceHandle& instance,
                               const vk::PhysicalDevice& physical_device)
    : instance(instance), physical_device(physical_device),
      supported_features(physical_device, instance), properties(physical_device, instance) {

    for (const auto& ext : physical_device.enumerateDeviceExtensionProperties()) {
        supported_extensions.emplace(ext.extensionName);
    }

    // Get memory properties and extension properties
    physical_device_memory_properties = physical_device.getMemoryProperties2();
    physical_device_extension_properties = physical_device.enumerateDeviceExtensionProperties();
}

PhysicalDeviceHandle PhysicalDevice::create(const InstanceHandle& instance,
                                            const vk::PhysicalDevice& physical_device) {
    return PhysicalDeviceHandle(new PhysicalDevice(instance, physical_device));
}

} // namespace merian
