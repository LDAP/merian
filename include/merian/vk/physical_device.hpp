#pragma once

#include "merian/fwd.hpp"
#include "merian/vk/instance.hpp"
#include "merian/vk/utils/vulkan_features.hpp"
#include "merian/vk/utils/vulkan_properties.hpp"

#include <fmt/format.h>
#include <memory>
#include <unordered_set>

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

    // ----------------------------------------

    const InstanceHandle& get_instance() const {
        return instance;
    }

    // ----------------------------------------

    bool extension_supported(const std::string& name) const {
        return supported_extensions.contains(name);
    }

    const std::unordered_set<std::string>& get_supported_extensions() {
        return supported_extensions;
    }

    // ----------------------------------------

    // Get reference to VulkanProperties aggregate containing all property structs
    const VulkanProperties& get_properties() const {
        return properties;
    }

    const vk::PhysicalDeviceMemoryProperties2& get_memory_properties() const {
        return physical_device_memory_properties;
    }

    const std::vector<vk::ExtensionProperties>& get_extension_properties() const {
        return physical_device_extension_properties;
    }

    const vk::PhysicalDeviceLimits& get_device_limits() const {
        return properties.get_properties2().properties.limits;
    }

    // ----------------------------------------

    // Get reference to VulkanFeatures aggregate containing all feature structs
    const VulkanFeatures& get_supported_features() const {
        return supported_features;
    }

    // ----------------------------------------

  private:
    const InstanceHandle instance;
    const vk::PhysicalDevice physical_device;

    std::unordered_set<std::string> supported_extensions;

    VulkanFeatures supported_features;

    VulkanProperties properties;

    vk::PhysicalDeviceMemoryProperties2 physical_device_memory_properties;

    std::vector<vk::ExtensionProperties> physical_device_extension_properties;
};

using PhysicalDeviceHandle = std::shared_ptr<PhysicalDevice>;

} // namespace merian
