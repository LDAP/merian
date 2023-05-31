#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkRayQuery : public Extension {
  public:
    ExtensionVkRayQuery() : Extension("ExtensionVkRaytraceQuery") {
        physical_device_ray_query_features.rayQuery = VK_TRUE;
    }
    ~ExtensionVkRayQuery() {}
    std::vector<const char*> required_device_extension_names() const override {
        return {
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // intel doesn't have it pre 2015 (hd 520)
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_RAY_QUERY_EXTENSION_NAME,
        };
    }

    // LIFECYCLE
    void* on_create_device(void* const p_next) override {
        physical_device_ray_query_features.pNext = p_next;
        return &physical_device_ray_query_features;
    }

  private:
    vk::PhysicalDeviceRayQueryFeaturesKHR physical_device_ray_query_features;
};

} // namespace merian
