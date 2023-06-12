#pragma once

#include "merian/vk/raytrace/as_builder.hpp"

namespace merian {

/*
 * A builder for top-level accelerations structures (BLASs) for ray-tracing.
 *
 *
 * Best practices: (from https://developer.nvidia.com/blog/rtx-best-practices/)
 *
 * For TLAS, consider the PREFER_FAST_TRACE flag and perform only rebuilds. Often, this results in
 * best overall performance. The rationale is that making the TLAS as high quality as possible
 * regardless of the movement occurring in the scene is important and doesn’t cost too much.
 *
 * Don’t include sky geometry in TLAS. A skybox or skysphere would have an AABB that overlaps with
 * everything else and all rays would have to be tested against it. It’s more efficient to handle
 * sky shading in the miss shader rather than in the hit shader for the geometry representing the
 * sky.
 *
 * Example:
 * std::vector<vk::AccelerationStructureInstanceKHR> instances;
 * vk::AccelerationStructureInstanceKHR instance{
 *     merian::transform_identity(),
 *     0,
 *     0xFF,
 *     0,
 *     vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable,
 *     blas->get_acceleration_structure_device_address()};
 * instances.push_back(instance);
 *
 * auto tlas_builder = merian::TLASBuilder(context, alloc);
 * auto instances_buffer = tlas_builder.cmd_make_instances_buffer(cmd, instances);
 * auto as = tlas_builder.queue_build(instances.size(), instances_buffer);
 * tlas_build.get_cmds(cmd);
 * // submit...
 */
class TLASBuilder : public ASBuilder {

  private:
    struct PendingTLAS {
        vk::AccelerationStructureBuildGeometryInfoKHR build_info;
        uint32_t instance_count;
        vk::AccelerationStructureGeometryKHR geometry;
    };

  public:
    TLASBuilder(const SharedContext context, const ResourceAllocatorHandle allocator);

    // Create the buffer that holds the instances on the GPU.
    // The upload only happens after the command buffer is submitted.
    BufferHandle
    cmd_make_instances_buffer(const vk::CommandBuffer cmd,
                              const std::vector<vk::AccelerationStructureInstanceKHR>& instances) {
        BufferHandle buffer = allocator->createBuffer(
            cmd, instances,
            vk::BufferUsageFlagBits::eShaderDeviceAddress |
                vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            {}, NONE, 16);
        // Make sure the upload has finished
        const vk::MemoryBarrier barrier{vk::AccessFlagBits::eTransferWrite,
                                        vk::AccessFlagBits::eAccelerationStructureWriteKHR};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, 1,
                            &barrier, 0, nullptr, 0, nullptr);
        return buffer;
    }

    // Ensures a TLAS build has finished.
    void cmd_barrier(const vk::CommandBuffer cmd, vk::PipelineStageFlags dst_pipeline_stages) {
        const vk::MemoryBarrier barrier{vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                            vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                        vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                            vk::AccessFlagBits::eShaderRead};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                            dst_pipeline_stages, {}, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    // Build a TLAS from instances that are stored on the device.
    AccelerationStructureHandle
    queue_build(const uint32_t instance_count,
                const BufferHandle& instances,
                const vk::BuildAccelerationStructureFlagsKHR flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace) {
        // Note: For some reason using a host buffer here kills the GPU (without layer error) :/
        vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
            false, {instances->get_device_address()}};
        return queue_build(instance_count, instances_data, flags);
    }

    // Build a TLAS from instances that are stored on the device.
    AccelerationStructureHandle
    queue_build(const uint32_t instance_count,
                const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                const vk::BuildAccelerationStructureFlagsKHR flags =
                    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Update a TLAS from instances that are stored on the device.
    void queue_update(const uint32_t instance_count,
                      const BufferHandle& instances,
                      const AccelerationStructureHandle src_as,
                      const vk::BuildAccelerationStructureFlagsKHR flags =
                          vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace) {
        // Note: For some reason using a host buffer here kills the GPU (without layer error) :/
        vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
            false, {instances->get_device_address()}};
        queue_update(instance_count, instances_data, src_as, flags);
    }

    // Update a TLAS from instances that are stored on the device.
    //
    // Consider using queue_rebuild, since the rebuild is fast and updating may hurt raytracing
    // performance.
    void queue_update(const uint32_t instance_count,
                      const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                      const AccelerationStructureHandle src_as,
                      const vk::BuildAccelerationStructureFlagsKHR flags =
                          vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);

    // Rebuild a TLAS from instances that are stored on the device.
    //
    // Consider using queue_rebuild, since the rebuild is fast and updating may hurt raytracing
    // performance.
    void queue_rebuild(const uint32_t instance_count,
                       const BufferHandle& instances,
                       const AccelerationStructureHandle src_as,
                       const vk::BuildAccelerationStructureFlagsKHR flags =
                           vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace) {
        // Note: For some reason using a host buffer here kills the GPU (without layer error) :/
        vk::AccelerationStructureGeometryInstancesDataKHR instances_data{
            false, {instances->get_device_address()}};
        queue_rebuild(instance_count, instances_data, src_as, flags);
    }

    // Rebuild a TLAS from instances that are stored on the device.
    void queue_rebuild(const uint32_t instance_count,
                       const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                       const AccelerationStructureHandle src_as,
                       const vk::BuildAccelerationStructureFlagsKHR flags);

    // Note: This method does not insert a synchronization barrier. You must enure proper
    // synchronization before using the TLAS (you can use the helper cmd_barrier()).
    void get_cmds(const vk::CommandBuffer cmd);

  private:
    std::vector<PendingTLAS> pending;
    vk::DeviceSize pending_min_scratch_buffer = 0;
};

} // namespace merian
