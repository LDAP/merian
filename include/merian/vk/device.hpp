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
           std::ranges::forward_range auto&& features,
           void* p_next)
        requires std::same_as<std::ranges::range_value_t<decltype(features)>, FeatureHandle>
        : physical_device(physical_device),
          enabled_extensions(extensions.begin(), extensions.end()) {

        for (const auto& feature : get_all_features()) {
            enabled_features.emplace(feature->get_structure_type(), feature);
        }

        void* create_device_p_next = p_next;
        for (const auto& feature : features) {
            feature->set_pnext(create_device_p_next);
            create_device_p_next = feature->get_structure_ptr();
        }
        const vk::DeviceCreateInfo device_create_info{{}, queue_create_infos,  {}, extensions,
                                                      {}, create_device_p_next};
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
           std::ranges::forward_range auto&& features,
           void* p_next)
        requires std::same_as<std::ranges::range_value_t<decltype(features)>, FeatureHandle>
    {
        return DeviceHandle(
            new Device(physical_device, queue_create_infos, extensions, features, p_next));
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

    // Can be called with vk::PhysicalDeviceFeatures2 and any struct that extends it.
    template <typename Features = vk::PhysicalDeviceFeatures>
    const Features* get_enabled_features() const {
        const auto it = enabled_features.find(Features::structureType);
        if (it == enabled_features.end()) {
            throw std::runtime_error{fmt::format("{} not a known feature struct.",
                                                 vk::to_string(Features::structureType))};
        }

        return reinterpret_cast<const Features*>(it->second->get_structure_ptr());
    }

    template <> const vk::PhysicalDeviceFeatures* get_enabled_features() const {
        return &(get_enabled_features<vk::PhysicalDeviceFeatures2>()->features);
    }

    const FeatureHandle& get_enabled_features(const vk::StructureType s_type) const {
        const auto it = enabled_features.find(s_type);
        if (it == enabled_features.end()) {
            throw std::runtime_error{
                fmt::format("{} not a known feature struct.", vk::to_string(s_type))};
        }

        return enabled_features.at(s_type);
    }

  private:
    const PhysicalDeviceHandle physical_device;
    const std::unordered_set<std::string> enabled_extensions;

    vk::Device device;

    std::unordered_map<vk::StructureType, FeatureHandle> enabled_features;

    vk::PipelineCache pipeline_cache;
};

using DeviceHandle = std::shared_ptr<Device>;

} // namespace merian
