#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkRayTracingPipeline : public Extension {
  public:
    ExtensionVkRayTracingPipeline() : Extension("ExtensionVkRayTracingPipeline") {
        ray_tracing_pipeline_features.rayTracingPipeline = VK_TRUE;
    }
    ~ExtensionVkRayTracingPipeline() {}
    std::vector<const char*> required_device_extension_names() const override {
        return {
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // intel doesn't have it pre 2015 (hd 520)
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,     VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        };
    }

    // LIFECYCLE

    void on_physical_device_selected(const vk::PhysicalDevice& physical_device) override {
        vk::PhysicalDeviceProperties2KHR props2;
        props2.pNext = &ray_tracing_pipeline_properties;
        physical_device.getProperties2(&props2);
    }
    void* on_create_device(void* const p_next) override {
        ray_tracing_pipeline_features.pNext = p_next;
        return &ray_tracing_pipeline_features;
    }

  private:
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_pipeline_features;

  public:
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_pipeline_properties;
};

} // namespace merian
