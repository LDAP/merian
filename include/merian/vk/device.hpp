#pragma once

#include "merian/vk/physical_device.hpp"
#include "spdlog/spdlog.h"

#include <memory>

namespace merian {

class Device : public std::enable_shared_from_this<Device> {
  private:
    // features and extensions are checked for support and skipped if not available.
    Device(const PhysicalDeviceHandle& physical_device,
           const VulkanFeatures& features,
           const vk::ArrayProxyNoTemporaries<const char*>& additional_extensions,
           const vk::ArrayProxyNoTemporaries<const vk::DeviceQueueCreateInfo>& queue_create_infos,
           void* p_next)
        : physical_device(physical_device) {

        SPDLOG_DEBUG("create device");

        SPDLOG_DEBUG("...with features:");
        for (const auto& feature_struct_name : features.get_feature_struct_names()) {
            for (const auto& feature_name : features.get_feature_names(feature_struct_name)) {
                if (features.get_feature(feature_struct_name, feature_name)) {
                    if (physical_device->get_supported_features().get_feature(feature_struct_name,
                                                                              feature_name)) {
                        SPDLOG_DEBUG("{}/{}", feature_struct_name, feature_name);
                        enabled_features.set_feature(feature_struct_name, feature_name, true);
                    } else {
                        SPDLOG_WARN("{}/{} requested but not supported", feature_struct_name,
                                    feature_name);
                    }
                }
            }
        }

        SPDLOG_DEBUG("...with extensions:");
        const auto feature_extensions = enabled_features.get_required_extensions(
            physical_device->get_instance()->get_vk_api_version());
        std::vector<const char*> all_extensions;
        all_extensions.reserve(additional_extensions.size() + feature_extensions.size());
        for (const auto* const ext : additional_extensions) {
            if (!physical_device->extension_supported(ext)) {
                SPDLOG_WARN("{} requested but not supported!", ext);
                continue;
            }
            auto [_, inserted] = enabled_extensions.emplace(ext);
            if (inserted) {
                all_extensions.emplace_back(ext);
                SPDLOG_DEBUG(ext);
            }
        }
        for (const auto* const ext : feature_extensions) {
            assert(physical_device->extension_supported(ext));

            auto [_, inserted] = enabled_extensions.emplace(ext);
            if (inserted) {
                all_extensions.emplace_back(ext);
                SPDLOG_DEBUG(ext);
            }
        }

        void* p_next_chain = enabled_features.build_chain_for_device_creation(p_next);
        const vk::DeviceCreateInfo device_create_info{{}, queue_create_infos, {}, all_extensions,
                                                      {}, p_next_chain};

        device = physical_device->get_physical_device().createDevice(device_create_info);
        SPDLOG_DEBUG("device {} created", fmt::ptr(VkDevice(device)));

        SPDLOG_DEBUG("create pipeline cache");
        vk::PipelineCacheCreateInfo pipeline_cache_create_info{};
        pipeline_cache = device.createPipelineCache(pipeline_cache_create_info);
    }

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

    // ---------------------------------------------

    bool extension_enabled(const std::string& name) {
        return enabled_extensions.contains(name);
    }

    // ---------------------------------------------

    const VulkanFeatures& get_enabled_features() const {
        return enabled_features;
    }

  private:
    const PhysicalDeviceHandle physical_device;

    std::unordered_set<std::string> enabled_extensions;
    VulkanFeatures enabled_features;

    vk::Device device;

    vk::PipelineCache pipeline_cache;
};

using DeviceHandle = std::shared_ptr<Device>;

} // namespace merian
