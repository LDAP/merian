#pragma once

#include "vk/extension/extension.hpp"

class ExtensionVkRaytraceQuery : public Extension {
  public:
    ExtensionVkRaytraceQuery() {
        acceleration_structure_features.accelerationStructure = VK_TRUE;
        ray_query_features.pNext = &acceleration_structure_features;
        ray_query_features.rayQuery = VK_TRUE;
    }
    ~ExtensionVkRaytraceQuery() {}
    std::string name() const override {
        return "ExtensionVkRaytraceQuery";
    }
    std::vector<const char*> required_device_extension_names() const override {
        return {
            // ray query instead of ray pipeline
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, // intel doesn't have it pre 2015 (hd 520)
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,     VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,    VK_KHR_RAY_QUERY_EXTENSION_NAME,
        };
    }
    void* on_create_device(void* p_next) override {
        acceleration_structure_features.pNext = p_next;
        return &ray_query_features;
    }


  private:
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features;
    vk::PhysicalDeviceRayQueryFeaturesKHR ray_query_features;
};
