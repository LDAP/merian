#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>
namespace merian {

/*
 * Abstract builder/updater class for bottom-level accelerations structures (BLASs) top-level
 * accelerations structures (TLASs) and for ray-tracing.
 *
 * BLASs hold the geometry, while top-level accelerations structures instances bottom-level as using
 * transformation matrices.
 *
 * This builder holds a scratch buffer that is large enough for the largest AS.
 * The scratch buffer can be released by calling release(). This class must kept alive until the
 * build has finished.
 */
class ASBuilder {

  public:
    ASBuilder(const SharedContext context, const ResourceAllocatorHandle& allocator)
        : context(context), allocator(allocator) {

        // query min scratch buffer alignment
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties;
        vk::PhysicalDeviceProperties2KHR props2;
        props2.pNext = &acceleration_structure_properties;
        context->pd_container.physical_device.getProperties2(&props2);
        scratch_buffer_min_alignment =
            acceleration_structure_properties.minAccelerationStructureScratchOffsetAlignment;
    }

    virtual ~ASBuilder() {}

    /* Releases the scratch buffer. Call if you do not plan to build more ASs.
     * 
     * Make sure that the build has finished when calling this!
     */
    void release() {
        scratch_buffer.reset();
        current_scratch_buffer_size = 0;
    }

  protected:
    // Ensures the scratch buffer has min size `min_size`.
    // Do not call if a build is running/pending.
    void ensure_scratch_buffer(vk::DeviceSize min_size) {
        if (current_scratch_buffer_size >= min_size) {
            return;
        }
        scratch_buffer.reset();
        scratch_buffer = allocator->createScratchBuffer(min_size, scratch_buffer_min_alignment,
                                                        "ASBuilder scratch buffer");
        current_scratch_buffer_size = min_size;
    }

  protected:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;
    vk::DeviceSize scratch_buffer_min_alignment;

    // The current scratch buffer, can be nullptr
    BufferHandle scratch_buffer;
    // Helps to determine if the scratch buffer needs to be enlarged
    vk::DeviceSize current_scratch_buffer_size = 0;
};

} // namespace merian
