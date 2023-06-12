#include "merian/vk/raytrace/blas_builder.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>
namespace merian {

BLASBuilder::BLASBuilder(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ASBuilder(context, allocator) {}

AccelerationStructureHandle
BLASBuilder::queue_build(const uint32_t geometry_count,
                         const vk::AccelerationStructureGeometryKHR* p_geometry,
                         const vk::AccelerationStructureBuildRangeInfoKHR* const* pp_range_info,
                         const vk::BuildAccelerationStructureFlagsKHR build_flags) {

    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        {}, // filled out later, after we know the size
        {}, // filled out later, after we know the size
        geometry_count,
        p_geometry};

    // Put primitive counts in own vector
    std::vector<uint32_t> primitive_counts(geometry_count);
    std::transform(pp_range_info, pp_range_info + geometry_count, primitive_counts.begin(),
                   [&](auto& range) { return range->primitiveCount; });

    vk::AccelerationStructureBuildSizesInfoKHR size_info =
        context->device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, build_info, primitive_counts);

    pending_min_scratch_buffer = std::max(pending_min_scratch_buffer, size_info.buildScratchSize);
    AccelerationStructureHandle as = allocator->createAccelerationStructure(
        vk::AccelerationStructureTypeKHR::eBottomLevel, size_info);
    build_info.dstAccelerationStructure = *as;
    pending.emplace_back(build_info, pp_range_info);

    return as;
}

void BLASBuilder::queue_update(
    const uint32_t geometry_count,
    const vk::AccelerationStructureGeometryKHR* p_geometry,
    const vk::AccelerationStructureBuildRangeInfoKHR* const* pp_range_info,
    const AccelerationStructureHandle as,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {

    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eUpdate,
        *as,
        *as,
        geometry_count,
        p_geometry};

    pending_min_scratch_buffer =
        std::max(pending_min_scratch_buffer, as->get_size_info().updateScratchSize);
    pending.emplace_back(build_info, pp_range_info);
}

void BLASBuilder::queue_rebuild(
    const uint32_t geometry_count,
    const vk::AccelerationStructureGeometryKHR* p_geometry,
    const vk::AccelerationStructureBuildRangeInfoKHR* const* pp_range_info,
    const AccelerationStructureHandle as,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        *as,
        *as,
        geometry_count,
        p_geometry};

    pending_min_scratch_buffer =
        std::max(pending_min_scratch_buffer, as->get_size_info().buildScratchSize);
    pending.emplace_back(build_info, pp_range_info);
}

AccelerationStructureHandle BLASBuilder::queue_build(
    const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
    const std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*>& range_info,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {

    return queue_build(geometry.size(), geometry.data(), range_info.data(), build_flags);
}

void BLASBuilder::get_cmds(const vk::CommandBuffer& cmd, const EventHandle& signal_event) {
    ensure_scratch_buffer(pending_min_scratch_buffer);
    assert(scratch_buffer);

    // Since the scratch buffer is reused across builds, we need a barrier to ensure one
    // build is finished before starting the next one.
    const vk::MemoryBarrier scratch_barrier{vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                  vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                              vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                  vk::AccessFlagBits::eAccelerationStructureWriteKHR};
    for (uint32_t idx = 0; idx < pending.size(); idx++) {
        pending[idx].build_info.scratchData.deviceAddress = scratch_buffer->get_device_address();

        cmd.buildAccelerationStructuresKHR(1, &pending[idx].build_info, pending[idx].range_info);

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, 1,
                            &scratch_barrier, 0, nullptr, 0, nullptr);
    }

    // Barrier for TLAS build / compaction reads
    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                  vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                              vk::AccessFlagBits::eAccelerationStructureReadKHR};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, 1, &barrier,
                        0, nullptr, 0, nullptr);

    if (signal_event)
        cmd.setEvent(signal_event->get_event(),
                     vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR);

    pending.clear();
    pending_min_scratch_buffer = 0;
}

} // namespace merian
