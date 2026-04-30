#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/raytrace/as_builder.hpp"

namespace merian {

vk::AccelerationStructureBuildSizesInfoKHR
ASBuilder::get_size_info(const uint32_t instance_count,
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

    return context->get_device()->get_device().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, build_info, instance_count);
}

AccelerationStructureHandle
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
        context->get_device()->get_device().getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, build_info, instance_count);

    AccelerationStructureHandle tlas = allocator->create_acceleration_structure(
        vk::AccelerationStructureTypeKHR::eTopLevel, size_info);

    build_info.dstAccelerationStructure = *tlas;

    pending_tlas_total_scratch +=
        align_ceil(size_info.buildScratchSize, scratch_buffer_min_alignment);
    pending_tlas_build_infos.push_back(build_info);
    pending_tlas_geometries.push_back(top_as_geometry);
    pending_tlas_range_infos.push_back({instance_count, 0, 0, 0});
    pending_tlas.emplace_back(tlas, size_info.buildScratchSize);

    return tlas;
}

void ASBuilder::queue_update(
    const uint32_t instance_count,
    const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
    const AccelerationStructureHandle& src_as,
    const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
    const vk::BuildAccelerationStructureFlagsKHR flags) {
    assert(src_as->get_size() >= size_info.accelerationStructureSize);

    vk::AccelerationStructureGeometryKHR top_as_geometry{vk::GeometryTypeKHR::eInstances,
                                                         {instances_data}};
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eTopLevel,
        flags,
        vk::BuildAccelerationStructureModeKHR::eUpdate,
        *src_as,
        *src_as,
        top_as_geometry};

    pending_tlas_total_scratch +=
        align_ceil(size_info.updateScratchSize, scratch_buffer_min_alignment);
    pending_tlas_build_infos.push_back(build_info);
    pending_tlas_geometries.push_back(top_as_geometry);
    pending_tlas_range_infos.push_back({instance_count, 0, 0, 0});
    pending_tlas.emplace_back(src_as, size_info.updateScratchSize);
}

void ASBuilder::queue_build(const uint32_t instance_count,
                            const vk::AccelerationStructureGeometryInstancesDataKHR& instances_data,
                            const AccelerationStructureHandle& src_as,
                            const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                            const vk::BuildAccelerationStructureFlagsKHR flags) {
    assert(src_as->get_size() >= size_info.accelerationStructureSize);

    vk::AccelerationStructureGeometryKHR top_as_geometry{vk::GeometryTypeKHR::eInstances,
                                                         {instances_data}};
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eTopLevel,
        flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        {},
        *src_as,
        top_as_geometry};

    pending_tlas_total_scratch +=
        align_ceil(size_info.buildScratchSize, scratch_buffer_min_alignment);
    pending_tlas_build_infos.push_back(build_info);
    pending_tlas_geometries.push_back(top_as_geometry);
    pending_tlas_range_infos.push_back({instance_count, 0, 0, 0});
    pending_tlas.emplace_back(src_as, size_info.buildScratchSize);
}

void ASBuilder::get_cmds_tlas(const CommandBufferHandle& cmd, BufferHandle& scratch_buffer) {
    if (pending_tlas.empty()) {
        return;
    }

    ensure_scratch_buffer(pending_tlas_total_scratch, scratch_buffer);
    cmd->keep_until_pool_reset(scratch_buffer);

    const vk::DeviceAddress scratch_base = scratch_buffer->get_device_address();
    vk::DeviceSize scratch_offset = 0;

    std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_info_ptrs(
        pending_tlas.size());

    for (uint32_t idx = 0; idx < pending_tlas.size(); idx++) {
        pending_tlas_build_infos[idx].scratchData.deviceAddress = scratch_base + scratch_offset;
        scratch_offset += align_ceil(pending_tlas[idx].scratch_size, scratch_buffer_min_alignment);
        // Fixup geometry pointer (invalidated by vector growth during queue calls)
        pending_tlas_build_infos[idx].pGeometries = &pending_tlas_geometries[idx];
        range_info_ptrs[idx] = &pending_tlas_range_infos[idx];
        cmd->keep_until_pool_reset(pending_tlas[idx].as);
    }

    cmd->get_command_buffer().buildAccelerationStructuresKHR(pending_tlas_build_infos,
                                                             range_info_ptrs);

    pending_tlas.clear();
    pending_tlas_build_infos.clear();
    pending_tlas_geometries.clear();
    pending_tlas_range_infos.clear();
    pending_tlas_total_scratch = 0;
}

} // namespace merian
