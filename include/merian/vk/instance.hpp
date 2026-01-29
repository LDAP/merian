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

    const uint32_t& get_vk_api_version() const {
        return vk_api_version;
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

  private:
    const vk::Instance instance;
    const uint32_t vk_api_version;

    const std::unordered_set<std::string> enabled_layers;
    const std::unordered_set<std::string> enabled_extensions;
};

using InstanceHandle = std::shared_ptr<Instance>;

} // namespace merian
