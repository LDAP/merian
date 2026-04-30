#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/raytrace/as_builder.hpp"

#include <vector>

namespace merian {

vk::AccelerationStructureBuildSizesInfoKHR
ASBuilder::get_size_info(const vk::AccelerationStructureGeometryKHR* geometry,
                         const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                         const vk::BuildAccelerationStructureFlagsKHR build_flags,
                         const uint32_t geometry_count) {
    const vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        {}, // empty here, used to find out the size to create a AS
        {}, // empty here, used to find out the size to create a AS
        geometry_count,
        geometry};

    // Put primitive counts in own vector
    std::vector<uint32_t> primitive_counts(geometry_count);
    std::transform(range_info, range_info + geometry_count, primitive_counts.begin(),
                   [&](auto& range) { return range.primitiveCount; });

    return context->get_device()->get_device().getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, build_info, primitive_counts);
}

vk::AccelerationStructureBuildSizesInfoKHR
ASBuilder::get_size_info(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                         const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                         const vk::BuildAccelerationStructureFlagsKHR build_flags) {
    return get_size_info(geometry.data(), range_info.data(), build_flags, geometry.size());
}

AccelerationStructureHandle
ASBuilder::queue_build(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                       const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                       const vk::BuildAccelerationStructureFlagsKHR build_flags) {
    assert(geometry.size() == range_info.size());
    return queue_build(geometry.data(), range_info.data(), build_flags, geometry.size());
}

AccelerationStructureHandle
ASBuilder::queue_build(const vk::AccelerationStructureGeometryKHR* geometry,
                       const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                       const vk::BuildAccelerationStructureFlagsKHR build_flags,
                       const uint32_t geometry_count) {

    const vk::AccelerationStructureBuildSizesInfoKHR size_info =
        get_size_info(geometry, range_info, build_flags, geometry_count);

    const AccelerationStructureHandle as = allocator->create_acceleration_structure(
        vk::AccelerationStructureTypeKHR::eBottomLevel, size_info);

    queue_build(geometry, range_info, as, size_info, build_flags, geometry_count);

    return as;
}

void ASBuilder::queue_build(
    const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
    const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
    const AccelerationStructureHandle& as,
    const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {
    assert(geometry.size() == range_info.size());
    queue_build(geometry.data(), range_info.data(), as, size_info, build_flags, geometry.size());
}

void ASBuilder::queue_build(const vk::AccelerationStructureGeometryKHR* geometry,
                            const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                            const AccelerationStructureHandle& as,
                            const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                            const vk::BuildAccelerationStructureFlagsKHR build_flags,
                            const uint32_t geometry_count) {
    assert(as->get_size() >= size_info.accelerationStructureSize);

    pending_blas_build_infos.push_back({vk::AccelerationStructureTypeKHR::eBottomLevel,
                                        build_flags,
                                        vk::BuildAccelerationStructureModeKHR::eBuild,
                                        {},
                                        *as,
                                        geometry_count,
                                        geometry});
    pending_blas_range_infos.push_back(range_info);
    pending_blas_total_scratch +=
        align_ceil(size_info.buildScratchSize, scratch_buffer_min_alignment);
    pending_blas.emplace_back(as, size_info.buildScratchSize);
}

void ASBuilder::queue_update(
    const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
    const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
    const AccelerationStructureHandle& as,
    const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {
    queue_update(geometry.data(), range_info.data(), as, size_info, build_flags);
}

void ASBuilder::queue_update(const vk::AccelerationStructureGeometryKHR* geometry,
                             const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                             const AccelerationStructureHandle& as,
                             const vk::AccelerationStructureBuildSizesInfoKHR& size_info,
                             const vk::BuildAccelerationStructureFlagsKHR build_flags,
                             const uint32_t geometry_count) {
    assert(as->get_size() >= size_info.accelerationStructureSize);

    pending_blas_build_infos.push_back({vk::AccelerationStructureTypeKHR::eBottomLevel, build_flags,
                                        vk::BuildAccelerationStructureModeKHR::eUpdate, *as, *as,
                                        geometry_count, geometry});
    pending_blas_range_infos.push_back(range_info);
    pending_blas_total_scratch +=
        align_ceil(size_info.updateScratchSize, scratch_buffer_min_alignment);
    pending_blas.emplace_back(as, size_info.updateScratchSize);
}

void ASBuilder::get_cmds_blas(const CommandBufferHandle& cmd, BufferHandle& scratch_buffer) {
    if (pending_blas.empty())
        return;

    ensure_scratch_buffer(pending_blas_total_scratch, scratch_buffer);
    assert(scratch_buffer);
    cmd->keep_until_pool_reset(scratch_buffer);

    const vk::DeviceAddress scratch_base = scratch_buffer->get_device_address();
    vk::DeviceSize scratch_offset = 0;

    for (uint32_t idx = 0; idx < pending_blas.size(); idx++) {
        pending_blas_build_infos[idx].scratchData.deviceAddress = scratch_base + scratch_offset;
        scratch_offset += align_ceil(pending_blas[idx].scratch_size, scratch_buffer_min_alignment);
        cmd->keep_until_pool_reset(pending_blas[idx].as);
    }

    cmd->get_command_buffer().buildAccelerationStructuresKHR(pending_blas_build_infos,
                                                             pending_blas_range_infos);

    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                    vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                        vk::AccessFlagBits::eAccelerationStructureWriteKHR};
    cmd->barrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                 vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, barrier);

    pending_blas.clear();
    pending_blas_build_infos.clear();
    pending_blas_range_infos.clear();
    pending_blas_total_scratch = 0;
}

} // namespace merian
