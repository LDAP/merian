#pragma once

#include "vk/extension/extension.hpp"

namespace merian {

/**
 * @brief      This class describes an extension to access VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME.
 *
 * To be efficient, ray tracing requires organizing the geometry into an acceleration structure (AS) that will reduce
 * the number of ray-triangle intersection tests during rendering. This is typically implemented in hardware as a
 * hierarchical structure, but only two levels are exposed to the user: a single top-level acceleration structure (TLAS)
 * referencing any number of bottom-level acceleration structures (BLAS), up to the limit
 * VkPhysicalDeviceAccelerationStructurePropertiesKHR::maxInstanceCount. Typically, a BLAS corresponds to individual 3D
 * models within a scene, and a TLAS corresponds to an entire scene built by positioning (with 3-by-4 transformation
 * matrices) individual referenced BLASes.
 *
 * BLASes store the actual vertex data. They are built from one or more vertex buffers, each with its own transformation
 * matrix (separate from the TLAS matrices), allowing us to store multiple positioned models within a single BLAS. Note
 * that if an object is instantiated several times within the same BLAS, its geometry will be duplicated. This can be
 * particularly useful for improving performance on static, non-instantiated scene components (as a rule of thumb, the
 * fewer BLAS, the better).
 *
 * The TLAS will contain the object instances, each with its own transformation matrix and reference to a corresponding
 * BLAS. We will start with a single bottom-level AS and a top-level AS instancing it once with an identity transform.
 *
 * ~ quote from https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/
 */
class ExtensionVkAccelerationStructure : public Extension {
  public:
    ExtensionVkAccelerationStructure() : Extension("ExtensionVkAccelerationStructure") {
        acceleration_structure_features.accelerationStructure = VK_TRUE;
    }
    ~ExtensionVkAccelerationStructure() {}

    std::vector<const char*> required_device_extension_names() const override {
        return {
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        };
    }

    // LIFECYCLE

    void on_physical_device_selected(const vk::PhysicalDevice& physical_device) override {
        vk::PhysicalDeviceProperties2KHR props2;
        props2.pNext = &acceleration_structure_properties;
        physical_device.getProperties2(&props2);
    }
    void* on_create_device(void* const p_next) override {
        acceleration_structure_features.pNext = p_next;
        return &acceleration_structure_features;
    }

  private:
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR acceleration_structure_features;

  public:
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties;
};

} // namespace merian
