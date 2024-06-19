#pragma once

#include "merian/vk/extension/extension_vk_acceleration_structure.hpp"
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
 *
 * Best practices: (from
 * https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/)
 *
 * For TLAS, consider the PREFER_FAST_TRACE flag and perform only rebuilds.
 * Often, this results in best overall performance.
 * The rationale is that making the TLAS as high quality as possible regardless of the movement
 * occurring in the scene is important and doesn’t cost too much.
 *
 * For static BLASes, use the PREFER_FAST_TRACE flag.
 * For all BLASes that are built only one time, optimizing for best ray-trace performance is an easy
 * choice.
 *
 * For dynamic BLASes, choose between using the PREFER_FAST_TRACE or PREFER_FAST_BUILD flags, or
 * neither. For BLASes that are occasionally rebuilt or updated, the optimal build preference flag
 * depends on many factors. How much is built? How expensive are the ray traces? Can the build cost
 * be hidden by executing builds on async compute? To find the optimal solution for a specific case,
 * I recommend trying out different options.
 *
 * Note: You cannot use one builder for multiple frames in flight, because the scratch buffer might
 * be reallocated!
 */
class ASBuilder {

  public:
    ASBuilder(const SharedContext context, const ResourceAllocatorHandle& allocator)
        : context(context), allocator(allocator),
          scratch_buffer_min_alignment(
              context->get_extension<ExtensionVkAccelerationStructure>()->min_scratch_alignment()) {
    }

    virtual ~ASBuilder() {}

  protected:
    // Ensures the scratch buffer has min size `min_size`.
    void ensure_scratch_buffer(const vk::DeviceSize min_size, BufferHandle& scratch_buffer) {
        if (scratch_buffer && scratch_buffer->get_size() >= min_size) {
            return;
        }
        scratch_buffer.reset();
        scratch_buffer = allocator->createScratchBuffer(min_size, scratch_buffer_min_alignment,
                                                        "ASBuilder scratch buffer");
    }

  protected:
    const SharedContext context;
    const ResourceAllocatorHandle allocator;
    const vk::DeviceSize scratch_buffer_min_alignment;
};

} // namespace merian
