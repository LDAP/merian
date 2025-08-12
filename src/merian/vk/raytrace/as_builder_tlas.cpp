#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/raytrace/as_builder.hpp"

namespace merian {

HWAccelerationStructureHandle
ASBuilder::queue_build(const uint32_t instance_count,
                       const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                       const vk::BuildAccelerationStructureFlagsKHR flags) {
    vk::AccelerationStructureGeometryKHR top_as_geometry{vk::GeometryTypeKHR::eInstances,
                                                         {instances_data}};
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eTopLevel,
        flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        {}, // filled out later, after we know the size
        {}, // filled out later, after we know the size
        top_as_geometry};

    vk::AccelerationStructureBuildSizesInfoKHR size_info =
        context->device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, build_info, instance_count);

    HWAccelerationStructureHandle tlas = allocator->createAccelerationStructure(
        vk::AccelerationStructureTypeKHR::eTopLevel, size_info);

    pending_min_scratch_buffer = std::max(pending_min_scratch_buffer, size_info.buildScratchSize);

    build_info.dstAccelerationStructure = *tlas;

    pending_tlas_builds.emplace_back(build_info, instance_count, top_as_geometry, tlas);

    return tlas;
}

void ASBuilder::queue_update(
    const uint32_t instance_count,
    const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
    const HWAccelerationStructureHandle& src_as,
    const vk::BuildAccelerationStructureFlagsKHR flags) {
    vk::AccelerationStructureGeometryKHR top_as_geometry{vk::GeometryTypeKHR::eInstances,
                                                         {instances_data}};
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eTopLevel,
        flags,
        vk::BuildAccelerationStructureModeKHR::eUpdate,
        *src_as,
        *src_as,
        top_as_geometry};

    pending_min_scratch_buffer =
        std::max(pending_min_scratch_buffer, src_as->get_size_info().updateScratchSize);

    pending_tlas_builds.emplace_back(build_info, instance_count, top_as_geometry, src_as);
}

void ASBuilder::queue_build(const uint32_t instance_count,
                            const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                            const HWAccelerationStructureHandle& src_as,
                            const vk::BuildAccelerationStructureFlagsKHR flags) {
    vk::AccelerationStructureGeometryKHR top_as_geometry{vk::GeometryTypeKHR::eInstances,
                                                         {instances_data}};
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eTopLevel,
        flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        {},
        *src_as,
        top_as_geometry};

    pending_min_scratch_buffer =
        std::max(pending_min_scratch_buffer, src_as->get_size_info().buildScratchSize);

    pending_tlas_builds.emplace_back(build_info, instance_count, top_as_geometry, src_as);
}

void ASBuilder::get_cmds_tlas(const CommandBufferHandle& cmd,
                              BufferHandle& scratch_buffer,
                              const ProfilerHandle& profiler) {
    if (pending_tlas_builds.empty()) {
        return;
    }

    ensure_scratch_buffer(pending_min_scratch_buffer, scratch_buffer);

    vk::AccelerationStructureBuildRangeInfoKHR build_offset_info{0, 0, 0, 0};

    // Since the scratch buffer is reused across builds, we need a barrier to ensure one
    // build is finished before starting the next one.
    const vk::BufferMemoryBarrier scratch_barrier =
        scratch_buffer->buffer_barrier(vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                           vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                       vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                           vk::AccessFlagBits::eAccelerationStructureWriteKHR);
    cmd->keep_until_pool_reset(scratch_buffer);
    for (uint32_t pending_idx = 0; pending_idx < pending_tlas_builds.size(); pending_idx++) {
        MERIAN_PROFILE_SCOPE_GPU(profiler, cmd, fmt::format("TLAS build {:02}", pending_idx));

        pending_tlas_builds[pending_idx].build_info.scratchData.deviceAddress =
            scratch_buffer->get_device_address();
        // Reset the pointer here, since it may have been invalidated
        pending_tlas_builds[pending_idx].build_info.pGeometries =
            &pending_tlas_builds[pending_idx].geometry;
        build_offset_info.primitiveCount = pending_tlas_builds[pending_idx].instance_count;

        cmd->get_command_buffer().buildAccelerationStructuresKHR(
            pending_tlas_builds[pending_idx].build_info, &build_offset_info);
        cmd->keep_until_pool_reset(pending_tlas_builds[pending_idx].tlas);
        cmd->barrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                     vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, scratch_barrier);
    }

    pending_tlas_builds.clear();
    pending_min_scratch_buffer = 0;
}

} // namespace merian
