#pragma once

#include "merian/vk/raytrace/blas_builder.hpp"

#include <vector>
#include <vulkan/vulkan.hpp>
namespace merian {

BLASBuilder::BLASBuilder(const SharedContext context, const ResourceAllocatorHandle allocator)
    : ASBuilder(context, allocator) {}

AccelerationStructureHandle
BLASBuilder::add_blas(const uint32_t geometry_count,
                      const vk::AccelerationStructureGeometryKHR* p_geometry,
                      const vk::AccelerationStructureBuildRangeInfoKHR* const* pp_range_info,
                      const vk::BuildAccelerationStructureFlagsKHR build_flags) {

    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        build_flags,
        vk::BuildAccelerationStructureModeKHR::eBuild,
        {}, // filled out later
        {}, // filled out later
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
        size_info.accelerationStructureSize, vk::AccelerationStructureTypeKHR::eBottomLevel);
    build_info.dstAccelerationStructure = *as;
    pending.emplace_back(build_info, pp_range_info);

    return as;
}

AccelerationStructureHandle BLASBuilder::add_blas(
    const std::vector<vk::AccelerationStructureGeometryKHR>& geometry,
    const std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*>& range_info,
    const vk::BuildAccelerationStructureFlagsKHR build_flags) {

    return add_blas(geometry.size(), geometry.data(), range_info.data(), build_flags);
}

void BLASBuilder::build(vk::CommandBuffer& cmd) {
    ensure_scratch_buffer(pending_min_scratch_buffer);
    assert(scratch_buffer);

    for (uint32_t idx = 0; idx < pending.size(); idx++) {
        pending[idx].build_info.scratchData.deviceAddress = scratch_buffer->get_device_address();

        cmd.buildAccelerationStructuresKHR(1, &pending[idx].build_info,
                                           pending[idx].range_info);

        // Since the scratch buffer is reused across builds, we need a barrier to ensure one
        // build is finished before starting the next one.
        vk::MemoryBarrier barrier{vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                      vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                  vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                      vk::AccessFlagBits::eAccelerationStructureWriteKHR};
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, 1,
                            &barrier, 0, nullptr, 0, nullptr);
    }

    // TODO: Move this to the caller?
    vk::MemoryBarrier barrier{vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                  vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                              vk::AccessFlagBits::eAccelerationStructureReadKHR};
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                        vk::PipelineStageFlagBits::eAllCommands, {}, 1, &barrier, 0, nullptr, 0,
                        nullptr);

    pending.clear();
    pending_min_scratch_buffer = 0;
}

} // namespace merian
