#pragma once

#include "merian/utils/alignment.hpp"
#include "merian/vk/memory/resource_allocations.hpp"
#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/utils/profiler.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

/*
 * Builder / Updater class for bottom-level accelerations structures (BLASs) top-level
 * accelerations structures (TLASs) for ray-tracing.
 *
 * BLASs hold the geometry, while top-level accelerations structures instances bottom-level as using
 * transformation matrices.
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
 */
class ASBuilder {

  private:
    struct PendingAS {
        AccelerationStructureHandle as;
        vk::DeviceSize scratch_size;
    };

  public:
    ASBuilder(const ContextHandle& context, const ResourceAllocatorHandle& allocator)
        : context(context), allocator(allocator) {

        if (context->get_device()
                ->get_enabled_features()
                .get_acceleration_structure_features_khr()
                .accelerationStructure == VK_TRUE) {
            scratch_buffer_min_alignment = context->get_physical_device()
                                               ->get_properties()
                                               .get_acceleration_structure_properties_khr()
                                               .minAccelerationStructureScratchOffsetAlignment;
        } else {
            SPDLOG_ERROR("AccelerationStructure support is required.");
        }
    }

    // BLAS BUILDS
    // ---------------------------------------------------------------------------

    vk::AccelerationStructureBuildSizesInfoKHR
    get_size_info(const vk::AccelerationStructureGeometryKHR* geometry,
                  const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                  const vk::BuildAccelerationStructureFlagsKHR build_flags,
                  const uint32_t geometry_count);

    vk::AccelerationStructureBuildSizesInfoKHR
    get_size_info(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                  const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                  const vk::BuildAccelerationStructureFlagsKHR build_flags =
                      vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Queries the sizes, allocates a acceleration structure and enqueues a BLAS to build for the
    // next get_cmds(). Returns the acceleration structure.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info
    // (pointers need to remain valid)!
    [[nodiscard]] AccelerationStructureHandle
    queue_build(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                const vk::BuildAccelerationStructureFlagsKHR build_flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Queries the sizes, allocates a acceleration structure and enqueues a BLAS to build for the
    // next get_cmds(). Returns the acceleration structure.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    [[nodiscard]]
    AccelerationStructureHandle
    queue_build(const vk::AccelerationStructureGeometryKHR* geometry,
                const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                const vk::BuildAccelerationStructureFlagsKHR build_flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
                const uint32_t geometry_count = 1);

    // Enqueues a BLAS to be (re)build with the next get_cmds().
    //
    // The geometry_count and build_flags members must have the same value which was specified when
    // `as` was last built. The caller must pass the current `size_info` (from get_size_info) — its
    // buildScratchSize is used to size the scratch buffer, and its accelerationStructureSize is
    // asserted to fit into `as`.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    void queue_build(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                     const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                     const AccelerationStructureHandle& as,
                     const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                     const vk::BuildAccelerationStructureFlagsKHR build_flags);

    // Enqueues a BLAS to be (re)build with the next get_cmds().
    //
    // The geometry_count and build_flags members must have the same value which was specified when
    // `as` was last built. The caller must pass the current `size_info` (from get_size_info) — its
    // buildScratchSize is used to size the scratch buffer, and its accelerationStructureSize is
    // asserted to fit into `as`.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    void queue_build(const vk::AccelerationStructureGeometryKHR* geometry,
                     const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                     const AccelerationStructureHandle& as,
                     const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                     const vk::BuildAccelerationStructureFlagsKHR build_flags,
                     const uint32_t geometry_count = 1);

    // Enqueues a BLAS to be updated with the next get_cmds().
    //
    // The geometry_count and build_flags members must have the same value which was specified when
    // `as` was last built. The caller must pass the current `size_info` — its updateScratchSize is
    // used to size the scratch buffer, and its accelerationStructureSize is asserted to fit into
    // `as`. Note: You should call queue_build after many updates or major deformation.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    void queue_update(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                      const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                      const AccelerationStructureHandle& as,
                      const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                      const vk::BuildAccelerationStructureFlagsKHR build_flags);

    // Enqueues a BLAS to be updated with the next get_cmds().
    //
    // The geometry_count and build_flags members must have the same value which was specified when
    // `as` was last built. The caller must pass the current `size_info` — its updateScratchSize is
    // used to size the scratch buffer, and its accelerationStructureSize is asserted to fit into
    // `as`. Note: You should call queue_build after many updates or major deformation.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    void queue_update(const vk::AccelerationStructureGeometryKHR* geometry,
                      const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                      const AccelerationStructureHandle& as,
                      const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                      const vk::BuildAccelerationStructureFlagsKHR build_flags,
                      const uint32_t geometry_count = 1);

    // TLAS BUILDS
    // ---------------------------------------------------------------------------

    vk::AccelerationStructureBuildSizesInfoKHR
    get_size_info(const uint32_t instance_count,
                  const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                  const vk::BuildAccelerationStructureFlagsKHR flags);

    vk::AccelerationStructureBuildSizesInfoKHR
    get_size_info(const uint32_t instance_count,
                  const BufferHandle& instances,
                  const vk::BuildAccelerationStructureFlagsKHR flags =
                      vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace) {
        vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
            VK_FALSE, {instances->get_device_address()}};
        return get_size_info(instance_count, instances_data, flags);
    }

    // Queries the sizes, allocates a acceleration structure and builds a TLAS from instances that
    // are stored on the device.
    [[nodiscard]]
    AccelerationStructureHandle
    queue_build(const uint32_t instance_count,
                const BufferHandle& instances,
                const vk::BuildAccelerationStructureFlagsKHR flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace) {
        // Note: For some reason using a host buffer here kills the GPU (without layer error) :/
        vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
            VK_FALSE, {instances->get_device_address()}};
        return queue_build(instance_count, instances_data, flags);
    }

    // Queries the sizes, allocates a acceleration structure and builds a TLAS from instances that
    // are stored on the device.
    [[nodiscard]]
    AccelerationStructureHandle
    queue_build(const uint32_t instance_count,
                const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                const vk::BuildAccelerationStructureFlagsKHR flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Update a TLAS from instances that are stored on the device.
    //
    // The caller must pass the current `size_info` — its updateScratchSize is used to size the
    // scratch buffer, and its accelerationStructureSize is asserted to fit into `src_as`.
    void queue_update(const uint32_t instance_count,
                      const BufferHandle& instances,
                      const AccelerationStructureHandle& src_as,
                      const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                      const vk::BuildAccelerationStructureFlagsKHR flags =
                          vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace) {
        // Note: For some reason using a host buffer here kills the GPU (without layer error) :/
        vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
            VK_FALSE, {instances->get_device_address()}};
        queue_update(instance_count, instances_data, src_as, size_info, flags);
    }

    // Rebuild a TLAS from instances that are stored on the device.
    //
    // The instance_count and build_flags members must have the same value which was specified when
    // `as` was last built. The caller must pass the current `size_info` — its buildScratchSize is
    // used to size the scratch buffer, and its accelerationStructureSize is asserted to fit into
    // `src_as`.
    void queue_build(const uint32_t instance_count,
                     const BufferHandle& instances,
                     const AccelerationStructureHandle& src_as,
                     const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                     const vk::BuildAccelerationStructureFlagsKHR flags =
                         vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace) {
        // Note: For some reason using a host buffer here kills the GPU (without layer error) :/
        vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
            VK_FALSE, {instances->get_device_address()}};
        queue_build(instance_count, instances_data, src_as, size_info, flags);
    }

    // Rebuild a TLAS from instances that are stored on the device.
    //
    // The instance_count and build_flags members must have the same value which was specified when
    // `as` was last built. The caller must pass the current `size_info` — its buildScratchSize is
    // used to size the scratch buffer, and its accelerationStructureSize is asserted to fit into
    // `src_as`.
    void queue_build(const uint32_t instance_count,
                     const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                     const AccelerationStructureHandle& src_as,
                     const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                     const vk::BuildAccelerationStructureFlagsKHR flags);

    // Update a TLAS from instances that are stored on the device.
    //
    // The caller must pass the current `size_info` — its updateScratchSize is used to size the
    // scratch buffer, and its accelerationStructureSize is asserted to fit into `src_as`.
    //
    // Consider using queue_rebuild, since the rebuild is fast and updating may hurt raytracing
    // performance.
    void queue_update(const uint32_t instance_count,
                      const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                      const AccelerationStructureHandle& src_as,
                      const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                      const vk::BuildAccelerationStructureFlagsKHR flags =
                          vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // ---------------------------------------------------------------------------

    // The returned buffer is the scratch buffer for this build, which has to be kept alive while
    // the build is not finished.
    //
    // Provide a BufferHandle to a (optinally null) scratch_buffer. The scratch buffer is reused if
    // it is large enough else it is replaced with a larger one. The scratch buffer is kept alive on
    // the pool.
    //
    // This command inserts a barrier for the BLAS that are built.
    void get_cmds_blas(const CommandBufferHandle& cmd, BufferHandle& scratch_buffer);

    // Note: This method does not insert a synchronization barrier. You must enure proper
    // synchronization before using the TLAS (you can use the helper cmd_barrier()).
    //
    // Provide a BufferHandle to a (optinally null) scratch_buffer. The scratch buffer is reused if
    // it is large enough else it is replaced with a larger one. The scratch buffer is kept alive on
    // the pool.
    void get_cmds_tlas(const CommandBufferHandle& cmd, BufferHandle& scratch_buffer);

    // Provide a BufferHandle to a (optinally null) scratch_buffer. The scratch buffer is reused if
    // it is large enough else it is replaced with a larger one. The scratch buffer is kept alive on
    // the pool.
    void get_cmds(const CommandBufferHandle& cmd, BufferHandle& scratch_buffer) {
        {
            MERIAN_PROFILE_SCOPE_GPU(cmd, "BLAS build");
            get_cmds_blas(cmd, scratch_buffer);
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(cmd, "TLAS build");
            get_cmds_tlas(cmd, scratch_buffer);
        }
    }

  private:
    // Ensures the scratch buffer has min size `min_size`.
    void ensure_scratch_buffer(const vk::DeviceSize min_size, BufferHandle& scratch_buffer) {
        if (scratch_buffer && scratch_buffer->get_size() >= min_size) {
            return;
        }
        scratch_buffer.reset();
        scratch_buffer = allocator->create_scratch_buffer(min_size, scratch_buffer_min_alignment,
                                                          "ASBuilder scratch buffer");
    }

  private:
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;
    vk::DeviceSize scratch_buffer_min_alignment;

    std::vector<PendingAS> pending_blas;
    std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> pending_blas_build_infos;
    std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> pending_blas_range_infos;
    vk::DeviceSize pending_blas_total_scratch = 0;

    std::vector<PendingAS> pending_tlas;
    std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> pending_tlas_build_infos;
    std::vector<vk::AccelerationStructureGeometryKHR> pending_tlas_geometries;
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> pending_tlas_range_infos;
    vk::DeviceSize pending_tlas_total_scratch = 0;
};

} // namespace merian
