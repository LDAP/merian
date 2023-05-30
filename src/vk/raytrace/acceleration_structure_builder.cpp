#include "vk/raytrace/acceleration_structure_builder.hpp"
#include <numeric>

namespace merian {

AccelerationStructureBuilder::~AccelerationStructureBuilder() {
    for (auto& blas : vec_blas) {
        resource_allocator.destroy(blas);
    }
    resource_allocator.destroy(tlas);
    vec_blas.clear();
}

vk::AccelerationStructureKHR AccelerationStructureBuilder::getAccelerationStructure() const {
    return tlas.as;
}

vk::DeviceAddress AccelerationStructureBuilder::getBlasDeviceAddress(uint32_t blasId) {
    assert(size_t(blasId) < vec_blas.size());
    return vec_blas[blasId].get_acceleration_structure_device_address(device);
}

// Build BLAS

void AccelerationStructureBuilder::buildBLAS(const std::vector<BlasInput>& input,
                                             vk::BuildAccelerationStructureFlagsKHR flags) {
    const uint32_t input_size = input.size();
    CommandPool cmd_pool(device, queue.get_queue_family_index());
    vk::DeviceSize asTotalSize = 0;    // Memory size of all allocated BLAS
    uint32_t nbCompactions = 0;        // Nb of BLAS requesting compaction
    vk::DeviceSize maxScratchSize = 0; // Largest scratch size

    // Preparing the information for the acceleration build commands.
    std::vector<BuildAccelerationStructureInfo> build_infos(input_size);

    for (uint32_t i = 0; i < input_size; i++) {
        auto& build_info_i = build_infos[i].buildInfo;
        auto& size_info_i = build_infos[i].sizeInfo;

        // Filling partially the vk::AccelerationStructureBuildGeometryInfoKHR for querying the build sizes.
        // Other information will be filled in the createBlas (see #2)
        build_info_i = vk::AccelerationStructureBuildGeometryInfoKHR{vk::AccelerationStructureTypeKHR::eBottomLevel,
                                                                     input[i].flags | flags,
                                                                     vk::BuildAccelerationStructureModeKHR::eBuild,
                                                                     {},
                                                                     {},
                                                                     input[i].asGeometry};

        // Build range information
        build_infos[i].rangeInfo = input[i].asBuildOffsetInfo.data();

        // Finding sizes to create acceleration structures and scratch
        std::vector<uint32_t> maxPrimitiveCounts(input[i].asBuildOffsetInfo.size());
        for (std::size_t tt = 0; tt < input[i].asBuildOffsetInfo.size(); tt++)
            maxPrimitiveCounts[tt] = input[i].asBuildOffsetInfo[tt].primitiveCount; // Number of primitives/triangles

        size_info_i = device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice,
                                                                   build_info_i, maxPrimitiveCounts);

        // Extra info
        asTotalSize += size_info_i.accelerationStructureSize;
        maxScratchSize = std::max(maxScratchSize, size_info_i.buildScratchSize);
        nbCompactions += (build_info_i.flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction) ? 1 : 0;
    }

    // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
    Buffer scratchBuffer = resource_allocator.createScratchBuffer(maxScratchSize);
    vk::DeviceAddress scratchAddress = scratchBuffer.get_device_address(device);

    // Allocate a query pool for storing the needed size for every BLAS compaction.
    // TODO: Extract compaction completely
    std::optional<vk::QueryPool> queryPool = std::nullopt;
    if (nbCompactions > 0) // Is compaction requested?
    {
        assert(nbCompactions == input_size); // Don't allow mix of on/off compaction
        vk::QueryPoolCreateInfo qpci{{}, vk::QueryType::eAccelerationStructureCompactedSizeKHR, input_size};
        queryPool = device.createQueryPool(qpci);
    }

    // Batching creation/compaction of BLAS to allow staying in restricted amount of memory (when compacting)
    // TODO: IMPORTANT: do not use _wait instead use barrieres and submit all at once!
    std::vector<uint32_t> batch_indices; // Indices of the BLAS to create
    vk::DeviceSize batch_device_size = 0;
    const vk::DeviceSize batchLimit = 256'000'000; // 256 MB
    for (uint32_t idx = 0; idx < input_size; idx++) {
        batch_indices.push_back(idx);
        batch_device_size += build_infos[idx].sizeInfo.accelerationStructureSize;
        // Over the limit or last BLAS element
        if (batch_device_size >= batchLimit || idx == input_size - 1) {

            // BUILD BATCH
            vk::CommandBuffer cmdBuf = cmd_pool.createCommandBuffer();
            cmdCreateBLAS(cmdBuf, batch_indices, build_infos, scratchAddress, queryPool);
            queue.submit_wait(cmdBuf);

            // COMPACT BATCH
            if (queryPool.has_value()) { // compact
                vk::CommandBuffer cmdBuf = cmd_pool.createCommandBuffer();
                cmdCompactBLAS(cmdBuf, batch_indices, build_infos, queryPool.value());
                queue.submit_wait(cmdBuf);
                // Destroy the non-compacted version
                destroyNonCompactedBLAS(batch_indices, build_infos);
            }

            // Reset
            batch_device_size = 0;
            batch_indices.clear();
        }
    }

    // Log reduction
#ifdef DEBUG
    if (queryPool.has_value()) {
        vk::DeviceSize compactSize =
            std::accumulate(build_infos.begin(), build_infos.end(), 0ULL,
                            [](const auto& a, const auto& b) { return a + b.sizeInfo.accelerationStructureSize; });
        const float fractionSmaller = (asTotalSize == 0) ? 0 : (asTotalSize - compactSize) / float(asTotalSize);
        SPDLOG_DEBUG("%scompated BLAS: reduced from: {} to: {}, ({}%)", asTotalSize, compactSize,
                     fractionSmaller * 100.f);
    }
#endif

    // Keeping all the created acceleration structures
    for (auto& b : build_infos) {
        vec_blas.emplace_back(b.as);
    }

    // Clean up
    if (queryPool.has_value())
        device.destroyQueryPool(queryPool.value());
    resource_allocator.finalizeAndReleaseStaging();
    resource_allocator.destroy(scratchBuffer);
}

void AccelerationStructureBuilder::cmdCreateBLAS(vk::CommandBuffer& cmdBuf,
                                                 std::vector<uint32_t>& indices,
                                                 std::vector<BuildAccelerationStructureInfo>& buildAs,
                                                 vk::DeviceAddress scratchAddress,
                                                 std::optional<vk::QueryPool>& queryPool) {
    if (queryPool.has_value()) // For querying the compaction size
        device.resetQueryPool(queryPool.value(), 0, static_cast<uint32_t>(indices.size()));
    uint32_t queryCnt = 0;

    for (const auto& idx : indices) {
        // Actual allocation of buffer and acceleration structure.
        buildAs[idx].as = resource_allocator.createAccelerationStructure(
            buildAs[idx].sizeInfo.accelerationStructureSize, vk::AccelerationStructureTypeKHR::eBottomLevel);

        // BuildInfo #2 part
        buildAs[idx].buildInfo.dstAccelerationStructure = buildAs[idx].as.as;
        // All build are using the same scratch buffer
        buildAs[idx].buildInfo.scratchData.deviceAddress = scratchAddress;

        // Building the bottom-level-acceleration-structure
        // TODO: Combine builds / multiple scratch buffer?
        cmdBuf.buildAccelerationStructuresKHR(1, &buildAs[idx].buildInfo, &buildAs[idx].rangeInfo);

        // Since the scratch buffer is reused across builds, we need a barrier to ensure one build
        // is finished before starting the next one.
        vk::MemoryBarrier barrier{vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                                  vk::AccessFlagBits::eAccelerationStructureWriteKHR};
        cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                               vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, 1, &barrier, 0, nullptr,
                               0, nullptr);

        if (queryPool.has_value()) {
            // Add a query to find the 'real' amount of memory needed, use for compaction
            cmdBuf.writeAccelerationStructuresPropertiesKHR(1, &buildAs[idx].buildInfo.dstAccelerationStructure,
                                                            vk::QueryType::eAccelerationStructureCompactedSizeKHR,
                                                            queryPool.value(), queryCnt++);
        }
    }
}

void AccelerationStructureBuilder::cmdCompactBLAS(vk::CommandBuffer& cmdBuf,
                                                  std::vector<uint32_t>& indices,
                                                  std::vector<BuildAccelerationStructureInfo>& buildAs,
                                                  vk::QueryPool& queryPool) {
    uint32_t queryCtn = 0;

    // Get the compacted size result back
    std::vector<vk::DeviceSize> compactSizes(static_cast<uint32_t>(indices.size()));
    check_result(device.getQueryPoolResults(queryPool, 0, (uint32_t)compactSizes.size(),
                                            compactSizes.size() * sizeof(vk::DeviceSize), compactSizes.data(),
                                            sizeof(VkDeviceSize), vk::QueryResultFlagBits::eWait),
                 "could not get query pool results");

    for (auto idx : indices) {
        buildAs[idx].cleanupAS = buildAs[idx].as;                                   // previous AS to destroy
        buildAs[idx].sizeInfo.accelerationStructureSize = compactSizes[queryCtn++]; // new reduced size

        // Creating a compact version of the AS
        buildAs[idx].as = resource_allocator.createAccelerationStructure(
            buildAs[idx].sizeInfo.accelerationStructureSize, vk::AccelerationStructureTypeKHR::eBottomLevel);

        // Copy the original BLAS to a compact version
        vk::CopyAccelerationStructureInfoKHR copy_info{buildAs[idx].buildInfo.dstAccelerationStructure,
                                                       buildAs[idx].as.as,
                                                       vk::CopyAccelerationStructureModeKHR::eCompact};
        cmdBuf.copyAccelerationStructureKHR(copy_info);
    }
}

void AccelerationStructureBuilder::destroyNonCompactedBLAS(std::vector<uint32_t>& indices,
                                                           std::vector<BuildAccelerationStructureInfo>& buildAs) {
    for (auto& i : indices) {
        resource_allocator.destroy(buildAs[i].cleanupAS);
    }
}

// UPDATE BLAS

void AccelerationStructureBuilder::updateBLAS(uint32_t blasIdx,
                                              BlasInput& blas,
                                              vk::BuildAccelerationStructureFlagsKHR flags) {
    assert(size_t(blasIdx) < vec_blas.size());

    // Preparing all build information, acceleration is filled later
    vk::AccelerationStructureBuildGeometryInfoKHR build_infos{
        vk::AccelerationStructureTypeKHR::eBottomLevel,
        flags,
        vk::BuildAccelerationStructureModeKHR::eUpdate,
        vec_blas[blasIdx].as,
        vec_blas[blasIdx].as,
        blas.asGeometry,
    };

    // Find size to build on the device
    std::vector<uint32_t> maxPrimCount(blas.asBuildOffsetInfo.size());
    for (std::size_t tt = 0; tt < blas.asBuildOffsetInfo.size(); tt++)
        maxPrimCount[tt] = blas.asBuildOffsetInfo[tt].primitiveCount; // Number of primitives/triangles

    vk::AccelerationStructureBuildSizesInfoKHR sizeInfo = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, build_infos, maxPrimCount);

    // Allocate the scratch buffer and setting the scratch info
    Buffer scratchBuffer = resource_allocator.createScratchBuffer(sizeInfo.buildScratchSize);
    build_infos.scratchData.deviceAddress = scratchBuffer.get_device_address(device);

    // Need pointer for some reason...
    std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> pBuildOffset(blas.asBuildOffsetInfo.size());
    for (size_t i = 0; i < blas.asBuildOffsetInfo.size(); i++)
        pBuildOffset[i] = &blas.asBuildOffsetInfo[i];

    // Update the instance buffer on the device side and build the TLAS
    CommandPool genCmdBuf(device, queue.get_queue_family_index());
    vk::CommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();

    // Update the acceleration structure. Note the VK_TRUE parameter to trigger the update,
    // and the existing BLAS being passed and updated in place
    cmdBuf.buildAccelerationStructuresKHR(build_infos, pBuildOffset);

    queue.submit_wait(cmdBuf);

    resource_allocator.destroy(scratchBuffer);
}

// BUILD TLAS

void AccelerationStructureBuilder::buildTLAS(const std::vector<vk::AccelerationStructureInstanceKHR>& instances,
                                             vk::BuildAccelerationStructureFlagsKHR flags,
                                             bool update) {
    assert(!tlas.as || update); // Cannot call buildTlas twice except to update.

    // Command buffer to create the TLAS
    CommandPool genCmdBuf(device, queue.get_queue_family_index());
    vk::CommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();

    // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
    // Buffer of instances containing the matrices and BLAS ids
    Buffer instancesBuffer =
        resource_allocator.createBuffer(cmdBuf, instances,
                                        vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR);

    vk::DeviceAddress instBufferAddr = instancesBuffer.get_device_address(device);

    // Make sure the instance buffer are copied before triggering the acceleration structure build
    vk::MemoryBarrier barrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eAccelerationStructureWriteKHR};
    cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                           vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, 1, &barrier, 0, nullptr, 0,
                           nullptr);

    // Creating the TLAS
    Buffer scratchBuffer;
    cmdCreateTLAS(cmdBuf, instances.size(), instBufferAddr, scratchBuffer, flags, update);

    // Finalizing and destroying temporary data
    queue.submit_wait(cmdBuf); // TODO: Do not use _wait (slow)
    resource_allocator.finalizeAndReleaseStaging();
    resource_allocator.destroy(scratchBuffer);
    resource_allocator.destroy(instancesBuffer);
}

void AccelerationStructureBuilder::cmdCreateTLAS(vk::CommandBuffer& cmdBuf,
                                                 uint32_t countInstance,
                                                 vk::DeviceAddress instBufferAddr,
                                                 Buffer& scratchBuffer,
                                                 vk::BuildAccelerationStructureFlagsKHR flags,
                                                 bool update) {
    // Wraps a device pointer to the above uploaded instances.
    vk::AccelerationStructureGeometryInstancesDataKHR instances_data{{}, {instBufferAddr}};

    // Put the above into a VkAccelerationStructureGeometryKHR. We need to put the instances struct in a union and label
    // it as instance data.
    vk::AccelerationStructureGeometryKHR top_as_geometry{vk::GeometryTypeKHR::eInstances, {instances_data}};

    // Find sizes
    vk::BuildAccelerationStructureModeKHR mode =
        update ? vk::BuildAccelerationStructureModeKHR::eBuild : vk::BuildAccelerationStructureModeKHR::eBuild;
    vk::AccelerationStructureBuildGeometryInfoKHR build_info{
        vk::AccelerationStructureTypeKHR::eTopLevel, flags, mode, {}, {}, top_as_geometry};

    vk::AccelerationStructureBuildSizesInfoKHR sizeInfo = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, build_info, countInstance);

    // Create TLAS
    if (!update) {
        tlas = resource_allocator.createAccelerationStructure(sizeInfo.accelerationStructureSize,
                                                              vk::AccelerationStructureTypeKHR::eTopLevel);
    }

    // Allocate the scratch memory
    scratchBuffer = resource_allocator.createScratchBuffer(sizeInfo.buildScratchSize);
    vk::DeviceAddress scratchAddress = scratchBuffer.get_device_address(device);

    // Update build information (only now possible after we know size)
    vk::AccelerationStructureKHR src_accel = update ? tlas.as : VK_NULL_HANDLE;
    build_info.srcAccelerationStructure = src_accel;
    build_info.dstAccelerationStructure = tlas.as;
    build_info.scratchData.deviceAddress = scratchAddress;

    // Build Offsets info: n instances
    vk::AccelerationStructureBuildRangeInfoKHR build_offset_info{countInstance, 0, 0, 0};

    // Build the TLAS
    cmdBuf.buildAccelerationStructuresKHR(build_info, &build_offset_info);
}

} // namespace merian
