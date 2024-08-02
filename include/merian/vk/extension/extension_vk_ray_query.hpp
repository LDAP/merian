#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * @brief      Adds support for VK_KHR_ray_query (and additional commonly required extensions).
 *
 * Allows tracing rays directly in compute shaders and graphics pipelines.
 * This extension requires ExtensionVkAccelerationStructure.
 */
class ExtensionVkRayQuery : public Extension {
  public:
    ExtensionVkRayQuery() : Extension("ExtensionVkRaytraceQuery") {
        physical_device_ray_query_features.rayQuery = VK_TRUE;
    }
    ~ExtensionVkRayQuery() {}
    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override {
        return {
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // intel doesn't have it pre 2015 (hd 520)
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,     VK_KHR_RAY_QUERY_EXTENSION_NAME,
        };
    }

    // LIFECYCLE
    void* pnext_device_create_info(void* const p_next) override {
        physical_device_ray_query_features.pNext = p_next;
        return &physical_device_ray_query_features;
    }

  private:
    vk::PhysicalDeviceRayQueryFeaturesKHR physical_device_ray_query_features;
};

} // namespace merian
