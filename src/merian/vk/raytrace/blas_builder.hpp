#pragma once

#include "merian/vk/command/event.hpp"
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
 *
 * Best practice: (from
 * https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/)
 *
 * Dynamic BLASes
 * Reuse the old BLAS when possible. Whenever you know that vertices of a BLAS have not moved after
 * the previous update, continue using the old BLAS.
 *
 * Update the BLAS only for visible objects. When instances are culled from the TLAS, also exclude
 * their culled BLASes from the BLAS update process.
 *
 * Consider skipping updates based on distance and size. Sometimes it’s not necessary to update a
 * BLAS on every frame, depending on how large it is on the screen. It may be possible to skip some
 * updates without causing noticeable visual errors.
 *
 * Rebuild BLASes after large deformations. BLAS updates are a good choice after limited
 * deformations, as they are significantly cheaper than rebuilds. However,large deformations after
 * the previous rebuild can lead to non-optimal ray-trace performance. Elongated triangles amplify
 * the issue.
 *
 * Consider rebuilding updated BLASes periodically. It can be non-trivial to detect when a geometry
 * has been deformed too much and would require a rebuild to restore optimal ray-trace performance.
 * Simply periodically rebuilding all BLASes can be a reasonable approach to avoid significant
 * performance implications, regardless of deformations.
 *
 * Distribute rebuilds over frames. Because rebuilds are considerably slower than updates, many
 * rebuilds on a single frame can lead to stuttering. To avoid this, it’s a good practice to
 * distribute the rebuilds over frames.
 *
 * Consider using only rebuilds with unpredictable deformations. In some cases, when the geometry
 * deformation is large and rapid enough, it’s beneficial to omit the ALLOW_UPDATE flag when
 * building the BLAS and always just rebuild it. If needed, using the PREFER_FAST_BUILD flag to
 * reduce the cost of rebuilding can be considered. In extreme cases, using the PREFER_FAST_BUILD
 * flag results in better overall ray-trace performance than using the PREFER_FAST_TRACE flag and
 * updating.
 *
 * Avoid triangle topology changes in BLAS updates. Topology changes in an update means that
 * triangles degenerate or revive. That can lead to non-optimal ray-trace performance if the
 * positions of the degenerate triangles do not represent the positions of the revived triangles.
 * Occasional topology changes in “bending” deformations are typically not problematic, but larger
 * topology changes in “breaking” deformations can be. When possible, prefer having separate BLAS
 * versions or using inactive triangles for different topologies caused by “breaking” deformations.
 * A triangle is inactive when its position is NaN. If those alternatives are not possible, I
 * recommend rebuilding the BLAS instead of updating after topology changes. Topology changes
 * through index buffer modifications are not allowed in updates.
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

    // Enqueues a BLAS to build for the next get_cmds().
    // Returns the acceleration structure. Note that you must keep the as alive and the structure is
    // only valid after the next build. You can free the pp_range_info and p_geometry after
    // get_cmds(). For static BLAS it is recommended to compact them afterwards.
    AccelerationStructureHandle
    queue_build(const uint32_t geometry_count,
                const vk::AccelerationStructureGeometryKHR* p_geometry,
                const vk::AccelerationStructureBuildRangeInfoKHR* const* pp_range_info,
                const vk::BuildAccelerationStructureFlagsKHR build_flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Enqueues a BLAS to build for the next get_cmds().
    // Returns the acceleration structure. Note that you must keep the as alive and the structure is
    // only valid after the next build. You can free the range_info and geometry after get_cmds()
    // (do not use initializer lists). For static BLAS it is recommended to compact them afterwards.
    AccelerationStructureHandle
    queue_build(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                const std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*>& range_info,
                const vk::BuildAccelerationStructureFlagsKHR build_flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Enqueues a BLAS to be updated with the next get_cmds().
    // Returns the acceleration structure. Note that you must keep the as alive and the structure is
    // only valid after the next build. You can free the pp_range_info and p_geometry after
    // get_cmds().
    // The geometry_count and build_flags members must have the same value which was specified when
    // `as` was last built. Note: You should call queue_rebuild after many updates or major
    // deformation.
    void queue_update(const uint32_t geometry_count,
                      const vk::AccelerationStructureGeometryKHR* p_geometry,
                      const vk::AccelerationStructureBuildRangeInfoKHR* const* pp_range_info,
                      const AccelerationStructureHandle as,
                      const vk::BuildAccelerationStructureFlagsKHR build_flags);

    // Enqueues a BLAS to be rebuild with the next get_cmds().
    // Returns the acceleration structure. Note that you must keep the as alive and the structure is
    // only valid after the next build. You can free the pp_range_info and p_geometry after
    // get_cmds().
    // The geometry_count and build_flags members must have the same value which was specified when
    // `as` was last built.
    void queue_rebuild(const uint32_t geometry_count,
                       const vk::AccelerationStructureGeometryKHR* p_geometry,
                       const vk::AccelerationStructureBuildRangeInfoKHR* const* pp_range_info,
                       const AccelerationStructureHandle as,
                       const vk::BuildAccelerationStructureFlagsKHR build_flags);

    // Note that you must execute the command buffer, else the returned acceleration structures are
    // not valid.
    void get_cmds(const vk::CommandBuffer& cmd, const EventHandle& compact_signal_event = nullptr);

  private:
    // The BLASs that are build when calling build()
    std::vector<PendingBLAS> pending;
    // The minimum scratch buffer size that is required to build all pending BLASs.
    vk::DeviceSize pending_min_scratch_buffer = 0;
};

} // namespace merian
