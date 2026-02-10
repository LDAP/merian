#pragma once

#include "merian/vk/physical_device.hpp"
#include "merian/vk/utils/vulkan_extensions.hpp"
#include "spdlog/spdlog.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace merian {

class Device : public std::enable_shared_from_this<Device> {
  private:
    // features and extensions are checked for support and skipped if not available.
    Device(const PhysicalDeviceHandle& physical_device,
           const VulkanFeatures& features,
           const vk::ArrayProxyNoTemporaries<const char*>& additional_extensions,
           const vk::ArrayProxyNoTemporaries<const vk::DeviceQueueCreateInfo>& queue_create_infos,
           void* p_next);

  public:
    static DeviceHandle
    create(const PhysicalDeviceHandle& physical_device,
           const VulkanFeatures& features,
           const vk::ArrayProxyNoTemporaries<const char*>& user_extensions,
           const vk::ArrayProxyNoTemporaries<const vk::DeviceQueueCreateInfo>& queue_create_infos,
           void* p_next) {
        return DeviceHandle(
            new Device(physical_device, features, user_extensions, queue_create_infos, p_next));
    }

    ~Device();

    const vk::PipelineCache& get_pipeline_cache() const {
        return pipeline_cache;
    }

    const vk::Device& get_device() const {
        return device;
    }

    const vk::Device& operator*() const {
        return device;
    }

    operator const vk::Device&() const {
        return device;
    }

    const PhysicalDeviceHandle& get_physical_device() const {
        return physical_device;
    }

    // ---------------------------------------------

    bool extension_enabled(const std::string& name) {
        return enabled_extensions.contains(name);
    }

    const std::unordered_set<std::string>& get_enabled_extensions() {
        return enabled_extensions;
    }

    const VulkanFeatures& get_enabled_features() const {
        return enabled_features;
    }

    const std::unordered_set<std::string>& get_enabled_spirv_extensions() const;

    const std::unordered_set<std::string>& get_enabled_spirv_capabilities() const;

    const std::map<std::string, std::string>& get_shader_defines() const;

    // Shortcut for get_physical_device()->get_vk_api_version()
    // Returns the effective API version of the physical device, that is the minimum of the
    // targeted version and the supported version.
    uint32_t get_vk_api_version() const {
        return physical_device->get_vk_api_version();
    }

    // ---------------------------------------------

    vk::PipelineStageFlags get_supported_pipeline_stages() const {
        return supported_pipeline_stages;
    }

    vk::PipelineStageFlags2 get_supported_pipeline_stages2() const {
        return supported_pipeline_stages2;
    }

  private:
    const PhysicalDeviceHandle physical_device;

    std::unordered_set<std::string> enabled_extensions;
    VulkanFeatures enabled_features;

    vk::Device device;
    vk::PipelineCache pipeline_cache;

    vk::PipelineStageFlags supported_pipeline_stages;
    vk::PipelineStageFlags2 supported_pipeline_stages2;

    std::unordered_set<std::string> enabled_spirv_extensions;
    std::unordered_set<std::string> enabled_spirv_capabilities;

    std::map<std::string, std::string> shader_defines;
};

using DeviceHandle = std::shared_ptr<Device>;

} // namespace merian
