#pragma once

#include "merian/fwd.hpp"

#include <memory>
#include <unordered_set>

// attempt to set the dynamic dispach launcher as early as possible
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif

#include <vulkan/vulkan.hpp>

namespace merian {

class Instance : public std::enable_shared_from_this<Instance> {
  private:
    Instance(const vk::InstanceCreateInfo& instance_create_info);

  public:
    // Returns the maximum API version the instance supports.
    static uint32_t get_instance_vk_api_version() {
        // See https://docs.vulkan.org/refpages/latest/refpages/source/VkApplicationInfo.html
        // Because Vulkan 1.0 implementations may fail with VK_ERROR_INCOMPATIBLE_DRIVER,
        // applications should determine the version of Vulkan available before calling
        // vkCreateInstance. If the vkGetInstanceProcAddr returns NULL for
        // vkEnumerateInstanceVersion, it is a Vulkan 1.0 implementation. Otherwise, the application
        // can call vkEnumerateInstanceVersion to determine the version of Vulkan.

        if (VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateInstanceVersion) {
            return vk::enumerateInstanceVersion();
        }
        return VK_API_VERSION_1_0;
    }

    static InstanceHandle create(const vk::InstanceCreateInfo& instance_create_info);

    ~Instance();

    const vk::Instance& get_instance() const {
        return instance;
    }

    const vk::Instance& operator*() const {
        return instance;
    }

    operator const vk::Instance&() const {
        return instance;
    }

    // Returns the effective API version of the instance, that is the minimium of the targeted
    // version and the supported instance version.
    const uint32_t& get_vk_api_version() const {
        return effective_vk_api_version;
    }

    // Returns the applications targeted API version. The effective
    // version for instance use (get_vk_api_version) might be lower.
    const uint32_t& get_target_vk_api_version() const {
        return target_vk_api_version;
    }

    const std::unordered_set<std::string>& get_enabled_layers() const {
        return enabled_layers;
    }

    const std::unordered_set<std::string>& get_enabled_extensions() const {
        return enabled_extensions;
    }

    bool layer_enabled(const std::string& layer) const {
        return enabled_layers.contains(layer);
    }

    bool extension_enabled(const std::string& extension) const {
        return enabled_extensions.contains(extension);
    }

    // this also queries all extensions, features and such, can be expensive to call!
    std::vector<PhysicalDeviceHandle> get_physical_devices();

    // --------------------------------------

    PFN_vkGetInstanceProcAddr get_vkGetInstanceProcAddr() const {
        return vk_get_instance_proc_addr;
    }

  private:
    const vk::Instance instance;
    const uint32_t effective_vk_api_version;
    const uint32_t target_vk_api_version;

    const std::unordered_set<std::string> enabled_layers;
    const std::unordered_set<std::string> enabled_extensions;

    PFN_vkGetInstanceProcAddr vk_get_instance_proc_addr;
};

using InstanceHandle = std::shared_ptr<Instance>;

} // namespace merian
