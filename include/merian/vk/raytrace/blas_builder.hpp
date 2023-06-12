#pragma once

#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/raytrace/as_builder.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>
namespace merian {

/*
 * A builder for bottom-level accelerations structures (BLASs) for ray-tracing.
 *
 * BLASs hold the geometry, while top-level accelerations structures instances bottom-level as using
 * transformation matrices.
 */
class BLASBuilder : public ASBuilder {

  private:
    struct PendingBLAS {
        // src/dstAccelerationStructures and scratchData.deviceAddress are left empty until build
        vk::AccelerationStructureBuildGeometryInfoKHR build_info;
        const vk::AccelerationStructureBuildRangeInfoKHR* const* range_info;
    };

  public:
    BLASBuilder(const SharedContext context, const ResourceAllocatorHandle allocator);

    // Enqueues a BLAS for the next build.
    // Returns the acceleration structure. Note that you must keep the as alive and the structure is
    // only valid after the next build. You can free the pp_range_info and p_geometry after build()
    AccelerationStructureHandle
    add_blas(const uint32_t geometry_count,
             const vk::AccelerationStructureGeometryKHR* p_geometry,
             const vk::AccelerationStructureBuildRangeInfoKHR* const* pp_range_info,
             const vk::BuildAccelerationStructureFlagsKHR build_flags =
                 vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Enqueues a BLAS for the next build.
    // Returns the acceleration structure. Note that you must keep the as alive and the structure is
    // only valid after the next build. You can free the range_info and geometry after build() (do
    // not use initializer lists).
    AccelerationStructureHandle
    add_blas(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
             const std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*>& range_info,
             const vk::BuildAccelerationStructureFlagsKHR build_flags =
                 vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Note that you must execute the command buffer, else the returned acceleration structures are
    // not valid.
    void build(vk::CommandBuffer& cmd);

  private:
    // The BLASs that are build when calling build()
    std::vector<PendingBLAS> pending;
    // The minimum scratch buffer size that is required to build all pending BLASs.
    vk::DeviceSize pending_min_scratch_buffer = 0;
};

} // namespace merian
