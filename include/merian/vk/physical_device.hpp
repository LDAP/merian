#pragma once

#include "merian/fwd.hpp"
#include "merian/vk/instance.hpp"
#include "merian/vk/utils/vulkan_features.hpp"
#include "merian/vk/utils/vulkan_properties.hpp"

#include <fmt/format.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

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

    // Returns the effective API version of the physical device, that is the minimum of the
    // targeted version and the supported version.
    uint32_t get_vk_api_version() const {
        return properties.get_vk_api_version();
    }

    // Returns the physical device's supported API version. The effective
    // version for device use (get_vk_api_version) might be lower.
    uint32_t get_physical_device_vk_api_version() const {
        return properties.get_physical_device_vk_api_version();
    }

    // ----------------------------------------

    const std::vector<const char*>& get_supported_spirv_extensions() const;
    const std::vector<const char*>& get_supported_spirv_capabilities() const;

    std::map<std::string, std::string> get_shader_defines() const;

    // ----------------------------------------

  private:
    const InstanceHandle instance;
    const vk::PhysicalDevice physical_device;

    std::unordered_set<std::string> supported_extensions;

    VulkanProperties properties;
    VulkanFeatures supported_features;

    vk::PhysicalDeviceMemoryProperties2 physical_device_memory_properties;

    std::vector<vk::ExtensionProperties> physical_device_extension_properties;

    std::vector<const char*> supported_spirv_extensions;
    std::vector<const char*> supported_spirv_capabilities;
};

using PhysicalDeviceHandle = std::shared_ptr<PhysicalDevice>;

} // namespace merian
