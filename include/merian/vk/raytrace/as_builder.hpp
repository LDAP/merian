#pragma once

#include "merian/vk/extension/extension_vk_acceleration_structure.hpp"
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
 */
class ASBuilder {

  private:
    struct PendingBLAS {
        // src/dstAccelerationStructures and scratchData.deviceAddress are left empty until build
        AccelerationStructureHandle blas;
        vk::AccelerationStructureBuildGeometryInfoKHR build_info;
        const vk::AccelerationStructureBuildRangeInfoKHR* range_info;
    };

    struct PendingTLAS {
        vk::AccelerationStructureBuildGeometryInfoKHR build_info;
        uint32_t instance_count;
        vk::AccelerationStructureGeometryKHR geometry;
    };

  public:
    ASBuilder(const ContextHandle& context, const ResourceAllocatorHandle& allocator)
        : context(context), allocator(allocator) {
        const auto ext = context->get_extension<ExtensionVkAccelerationStructure>();
        if (ext) {
            scratch_buffer_min_alignment = ext->min_scratch_alignment();
        } else {
            SPDLOG_ERROR("ExtensionVkAccelerationStructure is required.");
        }
    }

    // BLAS BUILDS
    // ---------------------------------------------------------------------------

    // Enqueues a BLAS to build for the next get_cmds().
    // Returns the acceleration structure.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    [[nodiscard]]
    AccelerationStructureHandle
    queue_build(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                const vk::BuildAccelerationStructureFlagsKHR build_flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Enqueues a BLAS to build for the next get_cmds().
    // Returns the acceleration structure.
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
    // `as` was last built.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    void queue_build(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                     const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                     const AccelerationStructureHandle& as,
                     const vk::BuildAccelerationStructureFlagsKHR build_flags);

    // Enqueues a BLAS to be (re)build with the next get_cmds().
    //
    // The geometry_count and build_flags members must have the same value which was specified when
    // `as` was last built.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    void queue_build(const vk::AccelerationStructureGeometryKHR* geometry,
                     const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                     const AccelerationStructureHandle& as,
                     const vk::BuildAccelerationStructureFlagsKHR build_flags,
                     const uint32_t geometry_count = 1);

    // Enqueues a BLAS to be updated with the next get_cmds().
    //
    // The geometry_count and build_flags members must have the same value which was specified when
    // `as` was last built. Note: You should call queue_rebuild after many updates or major
    // deformation.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    void queue_update(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                      const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                      const AccelerationStructureHandle& as,
                      const vk::BuildAccelerationStructureFlagsKHR build_flags);

    // Enqueues a BLAS to be updated with the next get_cmds().
    //
    // The geometry_count and build_flags members must have the same value which was specified when
    // `as` was last built. Note: You should call queue_rebuild after many updates or major
    // deformation.
    //
    // You must wait until after calling get_cmds() to free the geometry and range_info (pointers
    // need to remain valid)!
    void queue_update(const vk::AccelerationStructureGeometryKHR* geometry,
                      const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                      const AccelerationStructureHandle& as,
                      const vk::BuildAccelerationStructureFlagsKHR build_flags,
                      const uint32_t geometry_count = 1);

    // TLAS BUILDS
    // ---------------------------------------------------------------------------

    // Build a TLAS from instances that are stored on the device.
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

    // Build a TLAS from instances that are stored on the device.
    [[nodiscard]]
    AccelerationStructureHandle
    queue_build(const uint32_t instance_count,
                const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                const vk::BuildAccelerationStructureFlagsKHR flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Update a TLAS from instances that are stored on the device.
    void queue_update(const uint32_t instance_count,
                      const BufferHandle& instances,
                      const AccelerationStructureHandle& src_as,
                      const vk::BuildAccelerationStructureFlagsKHR flags =
                          vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace) {
        // Note: For some reason using a host buffer here kills the GPU (without layer error) :/
        vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
            VK_FALSE, {instances->get_device_address()}};
        queue_update(instance_count, instances_data, src_as, flags);
    }

    // Rebuild a TLAS from instances that are stored on the device.
    //
    // The instance_count and build_flags members must have the same value which was specified when
    // `as` was last built.
    void queue_build(const uint32_t instance_count,
                     const BufferHandle& instances,
                     const AccelerationStructureHandle& src_as,
                     const vk::BuildAccelerationStructureFlagsKHR flags =
                         vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace) {
        // Note: For some reason using a host buffer here kills the GPU (without layer error) :/
        vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
            VK_FALSE, {instances->get_device_address()}};
        queue_build(instance_count, instances_data, src_as, flags);
    }

    // Rebuild a TLAS from instances that are stored on the device.
    //
    // The instance_count and build_flags members must have the same value which was specified when
    // `as` was last built.
    void queue_build(const uint32_t instance_count,
                     const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                     const AccelerationStructureHandle& src_as,
                     const vk::BuildAccelerationStructureFlagsKHR flags);

    // Update a TLAS from instances that are stored on the device.
    //
    // Consider using queue_rebuild, since the rebuild is fast and updating may hurt raytracing
    // performance.
    void queue_update(const uint32_t instance_count,
                      const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                      const AccelerationStructureHandle& src_as,
                      const vk::BuildAccelerationStructureFlagsKHR flags =
                          vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // ---------------------------------------------------------------------------

    // The returned buffer is the scratch buffer for this build, which has to be kept alive while
    // the build is not finished.
    //
    // Provide a BufferHandle to a (optinally null) scratch_buffer. The scratch buffer is reused if
    // it is large enough else it is replaced with a larger one. Make sure to keep the scratch
    // buffer alive while processing has not finished on the GPU.
    //
    // This command inserts a barrier for the BLAS that are built.
    void get_cmds_blas(const CommandBufferHandle& cmd,
                       BufferHandle& scratch_buffer,
                       const ProfilerHandle& profiler = nullptr);

    // Note: This method does not insert a synchronization barrier. You must enure proper
    // synchronization before using the TLAS (you can use the helper cmd_barrier()).
    //
    // Provide a BufferHandle to a (optinally null) scratch_buffer. The scratch buffer is reused if
    // it is large enough else it is replaced with a larger one. Make sure to keep the scratch
    // buffer alive while processing has not finished on the GPU.
    void get_cmds_tlas(const CommandBufferHandle& cmd,
                       BufferHandle& scratch_buffer,
                       const ProfilerHandle& profiler = nullptr);

    // Provide a BufferHandle to a (optinally null) scratch_buffer. The scratch buffer is reused if
    // it is large enough else it is replaced with a larger one. Make sure to keep the scratch
    // buffer alive while processing has not finished on the GPU.
    void get_cmds(const CommandBufferHandle& cmd,
                  BufferHandle& scratch_buffer,
                  const ProfilerHandle& profiler = nullptr) {
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "BLAS build");
            get_cmds_blas(cmd, scratch_buffer, profiler);
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, "TLAS build");
            get_cmds_tlas(cmd, scratch_buffer, profiler);
        }
    }

  private:
    // Ensures the scratch buffer has min size `min_size`.
    void ensure_scratch_buffer(const vk::DeviceSize min_size, BufferHandle& scratch_buffer) {
        if (scratch_buffer && scratch_buffer->get_size() >= min_size) {
            return;
        }
        scratch_buffer.reset();
        scratch_buffer = allocator->createScratchBuffer(min_size, scratch_buffer_min_alignment,
                                                        "ASBuilder scratch buffer");
    }

  private:
    const ContextHandle context;
    const ResourceAllocatorHandle allocator;
    vk::DeviceSize scratch_buffer_min_alignment;

    // The BLASs/TLASs that are build when calling get_cmds()
    std::vector<PendingBLAS> pending_blas_builds;
    std::vector<PendingTLAS> pending_tlas_builds;
    // The minimum scratch buffer size that is required to build all pending BLASs.
    vk::DeviceSize pending_min_scratch_buffer = 0;
};

} // namespace merian
