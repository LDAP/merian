#include "merian/vk/raytrace/blas_builder.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>
namespace merian {

BLASBuilder::BLASBuilder(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ASBuilder(context, allocator) {}

AccelerationStructureHandle
BLASBuilder::queue_build(const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
                         const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
                         const vk::BuildAccelerationStructureFlagsKHR build_flags) {
    assert(geometry.size() == range_info.size());
    return queue_build(geometry.data(), range_info.data(), build_flags, geometry.size());
}

AccelerationStructureHandle
BLASBuilder::queue_build(const vk::AccelerationStructureGeometryKHR* geometry,
                         const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                         const vk::BuildAccelerationStructureFlagsKHR build_flags,
                         const uint32_t geometry_count) {

    // 1. Query the size of the AS to build
    //--------------------------------------------
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

    const vk::AccelerationStructureBuildSizesInfoKHR size_info =
        context->device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, build_info, primitive_counts);

    pending_min_scratch_buffer = std::max(pending_min_scratch_buffer, size_info.buildScratchSize);

    // 2. Create the AS with the aquired info
    //--------------------------------------------
    const AccelerationStructureHandle as = allocator->createAccelerationStructure(
        vk::AccelerationStructureTypeKHR::eBottomLevel, size_info);

    // 3. Enqueue the build with the new AS as target
    //--------------------------------------------
    queue_build(geometry, range_info, as, build_flags, geometry_count);

    return as;
}

void BLASBuilder::queue_build(
    const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
    const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
    const AccelerationStructureHandle& as,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {
    assert(geometry.size() == range_info.size());
    queue_build(geometry.data(), range_info.data(), as, build_flags, geometry.size());
}

void BLASBuilder::queue_build(const vk::AccelerationStructureGeometryKHR* geometry,
                              const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                              const AccelerationStructureHandle& as,
                              const vk::BuildAccelerationStructureFlagsKHR build_flags,
                              const uint32_t geometry_count) {
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        {},
        *as,
        geometry_count,
        geometry};

    pending_min_scratch_buffer =
        std::max(pending_min_scratch_buffer, as->get_size_info().buildScratchSize);
    pending.emplace_back(build_info, range_info);
}

void BLASBuilder::queue_update(
    const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
    const std::vector<vk::AccelerationStructureBuildRangeInfoKHR>& range_info,
    const AccelerationStructureHandle& as,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {
    queue_update(geometry.data(), range_info.data(), as, build_flags);
}

void BLASBuilder::queue_update(const vk::AccelerationStructureGeometryKHR* geometry,
                               const vk::AccelerationStructureBuildRangeInfoKHR* range_info,
                               const AccelerationStructureHandle& as,
                               const vk::BuildAccelerationStructureFlagsKHR build_flags,
                               const uint32_t geometry_count) {
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eUpdate,
        *as,
        *as,
        geometry_count,
        geometry};

    pending_min_scratch_buffer =
        std::max(pending_min_scratch_buffer, as->get_size_info().updateScratchSize);
    pending.emplace_back(build_info, range_info);
}

void BLASBuilder::get_cmds(const vk::CommandBuffer& cmd, BufferHandle& scratch_buffer) {
    if (pending.empty())
        return;

    ensure_scratch_buffer(pending_min_scratch_buffer, scratch_buffer);
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
        // Vulkan allows to create multiple as at once, however then the scratch buffer cannot be
        // reused! (This is the reason why we need to supply a pointer to a pointer for range
        // infos...)
        cmd.buildAccelerationStructuresKHR(1, &pending[idx].build_info, &pending[idx].range_info);
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

    pending.clear();
    pending_min_scratch_buffer = 0;
}

} // namespace merian
