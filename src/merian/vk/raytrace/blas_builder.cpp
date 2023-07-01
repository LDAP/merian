#include "merian/vk/raytrace/blas_builder.hpp"
#include "merian/vk/raytrace/as_compressor.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>
namespace merian {

BLASBuilder::BLASBuilder(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ASBuilder(context, allocator) {}

AccelerationStructureHandle
BLASBuilder::queue_build(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                         const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                         const vk::BuildAccelerationStructureFlagsKHR build_flags) {

    const uint32_t geometry_count = geometry.size();
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        {}, // filled out later, after we know the size
        {}, // filled out later, after we know the size
        geometry_count,
        geometry.data()};

    // Put primitive counts in own vector
    std::vector<uint32_t> primitive_counts(geometry_count);
    std::transform(range_info.begin(), range_info.end(), primitive_counts.begin(),
                   [&](auto& range) { return range.primitiveCount; });

    vk::AccelerationStructureBuildSizesInfoKHR size_info =
        context->device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, build_info, primitive_counts);

    pending_min_scratch_buffer = std::max(pending_min_scratch_buffer, size_info.buildScratchSize);
    AccelerationStructureHandle as = allocator->createAccelerationStructure(
        vk::AccelerationStructureTypeKHR::eBottomLevel, size_info);
    build_info.dstAccelerationStructure = *as;
    pending.emplace_back(build_info, geometry, range_info);

    return as;
}

void BLASBuilder::queue_update(
    const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
    const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
    const AccelerationStructureHandle as,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {

    const uint32_t geometry_count = geometry.size();
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eUpdate,
        *as,
        *as,
        geometry_count,
        geometry.data()};

    pending_min_scratch_buffer =
        std::max(pending_min_scratch_buffer, as->get_size_info().updateScratchSize);
    pending.emplace_back(build_info, geometry, range_info);
}

void BLASBuilder::queue_rebuild(
    const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
    const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
    const AccelerationStructureHandle as,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {

    const uint32_t geometry_count = geometry.size();
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        *as,
        *as,
        geometry_count,
        geometry.data()};

    pending_min_scratch_buffer =
        std::max(pending_min_scratch_buffer, as->get_size_info().buildScratchSize);
    pending.emplace_back(build_info, geometry, range_info);
}

void BLASBuilder::get_cmds(const vk::CommandBuffer& cmd, const EventHandle& compact_signal_event) {
    if (pending.empty())
        return;

    ensure_scratch_buffer(pending_min_scratch_buffer);
    assert(scratch_buffer);

    // Since the scratch buffer is reused across builds, we need a barrier to ensure one
    // build is finished before starting the next one.
    const vk::BufferMemoryBarrier scratch_barrier =
        scratch_buffer->buffer_barrier(vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                           vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                       vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                           vk::AccessFlagBits::eAccelerationStructureWriteKHR);
    for (uint32_t idx = 0; idx < pending.size(); idx++) {
        pending[idx].build_info.scratchData.deviceAddress = scratch_buffer->get_device_address();
        pending[idx].build_info.pGeometries = pending[idx].geometry.data();

        const vk::AccelerationStructureBuildRangeInfoKHR* p_range_info =
            pending[idx].range_info.data();
        cmd.buildAccelerationStructuresKHR(1, &pending[idx].build_info, &p_range_info);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, {},
                            scratch_barrier, {});
    }

    // Barrier for TLAS build / compaction reads
    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                        vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                    vk::AccessFlagBits::eAccelerationStructureReadKHR};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, 1, &barrier,
                        0, nullptr, 0, nullptr);

    if (compact_signal_event) {
        const vk::DependencyInfo dep_info{{}, ASCompressor::build_compress_barrier, {}, {}};
        cmd.setEvent2(compact_signal_event->get_event(), dep_info);
    }

    pending.clear();
    pending_min_scratch_buffer = 0;
}

} // namespace merian
