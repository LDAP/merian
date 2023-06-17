#include "merian/vk/raytrace/tlas_builder.hpp"

namespace merian {

TLASBuilder::TLASBuilder(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ASBuilder(context, allocator) {}

AccelerationStructureHandle
TLASBuilder::queue_build(const uint32_t instance_count,
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

    AccelerationStructureHandle tlas = allocator->createAccelerationStructure(
        vk::AccelerationStructureTypeKHR::eTopLevel, size_info);

    pending_min_scratch_buffer = std::max(pending_min_scratch_buffer, size_info.buildScratchSize);

    build_info.dstAccelerationStructure = *tlas;

    pending.emplace_back(build_info, instance_count, top_as_geometry);

    return tlas;
}

void TLASBuilder::queue_update(
    const uint32_t instance_count,
    const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
    const AccelerationStructureHandle src_as,
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

    pending.emplace_back(build_info, instance_count, top_as_geometry);
}

void TLASBuilder::queue_rebuild(
    const uint32_t instance_count,
    const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
    const AccelerationStructureHandle src_as,
    const vk::BuildAccelerationStructureFlagsKHR flags) {
    vk::AccelerationStructureGeometryKHR top_as_geometry{vk::GeometryTypeKHR::eInstances,
                                                         {instances_data}};
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eTopLevel,
        flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        *src_as,
        *src_as,
        top_as_geometry};

    pending_min_scratch_buffer =
        std::max(pending_min_scratch_buffer, src_as->get_size_info().buildScratchSize);

    pending.emplace_back(build_info, instance_count, top_as_geometry);
}

void TLASBuilder::get_cmds(const vk::CommandBuffer cmd) {
    ensure_scratch_buffer(pending_min_scratch_buffer);

    vk::AccelerationStructureBuildRangeInfoKHR build_offset_info{0, 0, 0, 0};

    for (uint32_t pending_idx = 0; pending_idx < pending.size(); pending_idx++) {
        pending[pending_idx].build_info.scratchData.deviceAddress =
            scratch_buffer->get_device_address();
        // Reset the pointer here, since it may have been invalidated
        pending[pending_idx].build_info.pGeometries = &pending[pending_idx].geometry;
        build_offset_info.primitiveCount = pending[pending_idx].instance_count;

        cmd.buildAccelerationStructuresKHR(pending[pending_idx].build_info, &build_offset_info);

        // Since the scratch buffer is reused across builds, we need a barrier to ensure one
        // build is finished before starting the next one.
        const vk::MemoryBarrier barrier{vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                            vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                        vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                            vk::AccessFlagBits::eAccelerationStructureWriteKHR};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, 1,
                            &barrier, 0, nullptr, 0, nullptr);
    }

    pending.clear();
    pending_min_scratch_buffer = 0;
}

} // namespace merian