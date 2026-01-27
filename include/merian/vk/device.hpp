#pragma once

#include "merian/vk/physical_device.hpp"
#include "spdlog/spdlog.h"

#include <memory>

namespace merian {

class Device : public std::enable_shared_from_this<Device> {
  private:
    Device(const PhysicalDeviceHandle& physical_device,
           const vk::ArrayProxyNoTemporaries<const vk::DeviceQueueCreateInfo>& queue_create_infos,
           const vk::ArrayProxyNoTemporaries<const char*>& extensions,
           const std::vector<std::string>& requested_feature_names,
           void* p_next)
        : physical_device(physical_device),
          enabled_extensions(extensions.begin(), extensions.end()),
          enabled_features(physical_device->get_supported_features()) {

        // Enable requested features based on feature names
        for (const auto& feature_name : requested_feature_names) {
            // Parse feature_name and enable via enabled_features.set_feature()
            // For now, we'll copy all supported features and assume they're enabled
            // TODO: Parse feature names properly
        }

        // Build pNext chain from enabled features
        void* feature_chain = enabled_features.build_chain_for_device_creation();

        // Chain with any additional p_next from caller
        if (p_next) {
            // Find the end of feature_chain and append p_next
            void** current = &feature_chain;
            while (*current) {
                current = reinterpret_cast<void**>(static_cast<char*>(*current) + sizeof(vk::StructureType));
            }
            *current = p_next;
        }

        const vk::DeviceCreateInfo device_create_info{{}, queue_create_infos,  {}, extensions,
                                                      {}, feature_chain};
        device = physical_device->get_physical_device().createDevice(device_create_info);

        SPDLOG_DEBUG("create pipeline cache");
        vk::PipelineCacheCreateInfo pipeline_cache_create_info{};
        pipeline_cache = device.createPipelineCache(pipeline_cache_create_info);
    }

  public:
    static DeviceHandle
    create(const PhysicalDeviceHandle& physical_device,
           const vk::ArrayProxyNoTemporaries<const vk::DeviceQueueCreateInfo>& queue_create_infos,
           const vk::ArrayProxyNoTemporaries<const char*>& extensions,
           const std::vector<std::string>& requested_feature_names = {},
           void* p_next = nullptr)
    {
        return DeviceHandle(
            new Device(physical_device, queue_create_infos, extensions, requested_feature_names, p_next));
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

    // ---------------------------------------------

    bool extension_enabled(const std::string& name) {
        return enabled_extensions.contains(name);
    }

    // ---------------------------------------------

    // Get reference to VulkanFeatures aggregate containing all enabled features
    const VulkanFeatures& get_enabled_features() const {
        return enabled_features;
    }

  private:
    const PhysicalDeviceHandle physical_device;
    const std::unordered_set<std::string> enabled_extensions;

    vk::Device device;

    VulkanFeatures enabled_features;

    vk::PipelineCache pipeline_cache;
};

using DeviceHandle = std::shared_ptr<Device>;

} // namespace merian
