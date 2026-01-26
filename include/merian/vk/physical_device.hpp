#pragma once

#include "merian/fwd.hpp"
#include "merian/vk/instance.hpp"
#include "merian/vk/utils/extensions.hpp"
#include "merian/vk/utils/features.hpp"

#include <fmt/format.h>
#include <memory>
#include <unordered_map>
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

    bool extension_supported(const std::string& name) {
        return supported_extensions.contains(name);
    }

    // ----------------------------------------

    // Can be called with vk::PhysicalDeviceProperties2 and any struct that extends it. If a stuct
    // type is not supported by this physical device, then an runtime_error is thown of nullptr_ok
    // == false, otherwise nullptr is returned.
    template <typename Properties = vk::PhysicalDeviceProperties>
    const Properties* get_properties(const bool nullptr_ok = false) const {
        const auto it = properties.find(Properties::structureType);
        if (it == properties.end()) {
            if (nullptr_ok) {
                return nullptr;
            }
            throw std::runtime_error{fmt::format("{} not in properties of this physical device.",
                                                 vk::to_string(Properties::structureType))};
        }

        return reinterpret_cast<const Properties*>(it->second->get_structure_ptr());
    }

    template <>
    const vk::PhysicalDeviceProperties2* get_properties(const bool /*nullptr_ok*/) const {
        return &physical_device_properties;
    }

    template <>
    const vk::PhysicalDeviceProperties* get_properties(const bool /*nullptr_ok*/) const {
        return &physical_device_properties.properties;
    }

    const vk::PhysicalDeviceMemoryProperties2& get_memory_properties() const {
        return physical_device_memory_properties;
    }

    const std::vector<vk::ExtensionProperties>& get_extension_properties() const {
        return physical_device_extension_properties;
    }

    const vk::PhysicalDeviceLimits& get_device_limits() const {
        return physical_device_properties.properties.limits;
    }

    // ----------------------------------------

    // Can be called with vk::PhysicalDeviceFeatures2 and any struct that extends it.
    template <typename Features = vk::PhysicalDeviceFeatures>
    const Features* get_supported_features() const {
        const auto it = supported_features.find(Features::structureType);
        if (it == supported_features.end()) {
            throw std::runtime_error{fmt::format("{} not a known feature struct.",
                                                 vk::to_string(Features::structureType))};
        }

        return reinterpret_cast<const Features*>(it->second->get_structure_ptr());
    }

    template <> const vk::PhysicalDeviceFeatures* get_supported_features() const {
        return &(get_supported_features<vk::PhysicalDeviceFeatures2>()->features);
    }

    const FeatureHandle& get_supported_features(const vk::StructureType s_type) const {
        const auto it = supported_features.find(s_type);
        if (it == supported_features.end()) {
            throw std::runtime_error{
                fmt::format("{} not a known feature struct.", vk::to_string(s_type))};
        }

        return supported_features.at(s_type);
    }

    // ----------------------------------------

  private:
    const InstanceHandle instance;
    const vk::PhysicalDevice physical_device;

    std::unordered_set<std::string> supported_extensions;

    vk::PhysicalDeviceProperties2 physical_device_properties;
    std::unordered_map<vk::StructureType, std::shared_ptr<Property>> properties;

    vk::PhysicalDeviceMemoryProperties2 physical_device_memory_properties;

    std::vector<vk::ExtensionProperties> physical_device_extension_properties;

    std::unordered_map<vk::StructureType, FeatureHandle> supported_features;
};

using PhysicalDeviceHandle = std::shared_ptr<PhysicalDevice>;

} // namespace merian
