#pragma once

#include "merian/fwd.hpp"
#include "merian/vk/instance.hpp"

#include <memory>

namespace merian {

class PhysicalDevice : public std::enable_shared_from_this<PhysicalDevice> {
  private:
    PhysicalDevice(const InstanceHandle& instance, const vk::PhysicalDevice& physical_device);

  public:
    static PhysicalDeviceHandle create(const InstanceHandle& instance,
                                       const vk::PhysicalDevice& physical_device);

    const vk::PhysicalDevice& get_physical_device() const {
        return physical_device;
    }

    const vk::PhysicalDevice& operator*() const {
        return physical_device;
    }

    operator const vk::PhysicalDevice&() const {
        return physical_device;
    }

    const vk::PhysicalDeviceLimits& get_physical_device_limits() const {
        return physical_device_properties.properties.limits;
    }

  public:
    const InstanceHandle instance;
    const vk::PhysicalDevice physical_device;

    vk::PhysicalDeviceProperties2 physical_device_properties;
    vk::PhysicalDeviceVulkan11Properties physical_device_11_properties;
    vk::PhysicalDeviceVulkan12Properties physical_device_12_properties;
    vk::PhysicalDeviceVulkan13Properties physical_device_13_properties;
    vk::PhysicalDeviceVulkan14Properties physical_device_14_properties;

    // all supported features. Must be enabled with ExtensionVkCore.
    vk::PhysicalDeviceFeatures2 physical_device_features;
    vk::PhysicalDeviceMemoryProperties2 physical_device_memory_properties;
    vk::PhysicalDeviceSubgroupProperties physical_device_subgroup_properties;
    vk::PhysicalDeviceSubgroupSizeControlProperties
        physical_device_subgroup_size_control_properties;
    std::vector<vk::ExtensionProperties> physical_device_extension_properties;
};

using PhysicalDeviceHandle = std::shared_ptr<PhysicalDevice>;

} // namespace merian
