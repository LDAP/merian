#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * @brief      Adds support for VK_KHR_ray_tracing_pipeline (and additional commonly required
 * extensions).
 *
 * Raytracing pipelines are implemented as multiple shaders that generate rays and process
 * intersections (callable shaders). This extension requires ExtensionVkAccelerationStructure.
 */
class ExtensionVkRayTracingPipeline : public Extension {
  public:
    ExtensionVkRayTracingPipeline() : Extension("ExtensionVkRayTracingPipeline") {}

    ~ExtensionVkRayTracingPipeline() {}

    std::vector<const char*>
    required_device_extension_names(vk::PhysicalDevice /*unused*/) const override {
        return {
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // intel doesn't have it pre 2015 (hd 520)
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        };
    }

    // LIFECYCLE

    void* pnext_get_features_2(void* const p_next) override {
        ray_tracing_pipeline_features.setPNext(p_next);
        return &ray_tracing_pipeline_features;
    }

    bool extension_supported(const Context::PhysicalDeviceContainer& /*unused*/) override {
        return ray_tracing_pipeline_features.rayTracingPipeline == VK_TRUE;
    }

    void
    on_physical_device_selected(const Context::PhysicalDeviceContainer& pd_container) override {
        vk::PhysicalDeviceProperties2KHR props2;
        props2.pNext = &ray_tracing_pipeline_properties;
        pd_container.physical_device.getProperties2(&props2);
    }

    void* pnext_device_create_info(void* const p_next) override {
        ray_tracing_pipeline_features.pNext = p_next;
        return &ray_tracing_pipeline_features;
    }

  private:
    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_pipeline_features;

  public:
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_pipeline_properties;
};

} // namespace merian
