#include "merian/vk/raytrace/as_compressor.hpp"

#include "merian/vk/utils/check_result.hpp"

namespace merian {

// You must ensure proper synchronization with the build
std::vector<AccelerationStructureHandle>
ASCompressor::compact(const SharedContext& context,
                      const ResourceAllocatorHandle& allocator,
                      const CommandPoolHandle& pool,
                      const QueueHandle& queue,
                      const std::vector<AccelerationStructureHandle>& ass,
                      const EventHandle& build_wait_event,
                      const vk::AccelerationStructureTypeKHR type) {
    vk::QueryPoolCreateInfo qpci{{},
                                 vk::QueryType::eAccelerationStructureCompactedSizeKHR,
                                 static_cast<uint32_t>(ass.size())};
    vk::QueryPool query_pool = context->device.createQueryPool(qpci);

    vk::CommandBuffer cmd = pool->create_and_begin();

    cmd.resetQueryPool(query_pool, 0, static_cast<uint32_t>(ass.size()));

    // Wait for build to complete
    const vk::DependencyInfo dep_info{{}, ASCompressor::build_compress_barrier, {}, {}};
    cmd.setEvent2(build_wait_event->get_event(), dep_info);

    // Query compacted size
    std::vector<vk::AccelerationStructureKHR> acc_structures(ass.size());
    std::transform(ass.begin(), ass.end(), acc_structures.begin(),
                   [&](auto& as) { return as->get_acceleration_structure(); });
    cmd.writeAccelerationStructuresPropertiesKHR(
        acc_structures, vk::QueryType::eAccelerationStructureCompactedSizeKHR, query_pool, 0);
    vk::Fence fence = context->device.createFence({});
    pool->end_all();
    queue->submit(pool, fence);

    // TODO: Can this be done without waiting?
    check_result(context->device.waitForFences(fence, true, ~0), "failed waiting for fences");
    pool->reset();
    context->device.resetFences(fence);
    cmd = pool->create_and_begin();

    std::vector<vk::DeviceSize> compact_sizes(static_cast<uint32_t>(ass.size()));
    check_result(context->device.getQueryPoolResults(query_pool, 0, (uint32_t)compact_sizes.size(),
                                                     compact_sizes.size() * sizeof(vk::DeviceSize),
                                                     compact_sizes.data(), sizeof(VkDeviceSize),
                                                     vk::QueryResultFlagBits::eWait),
                 "could not get query pool results");

    std::vector<AccelerationStructureHandle> result(ass.size());
    for (uint32_t i = 0; i < ass.size(); i++) {

        auto size_info = ass[i]->get_size_info();
        size_info.accelerationStructureSize = compact_sizes[i];

        // Creating a compact version of the AS
        result[i] = allocator->createAccelerationStructure(type, size_info);

        // Copy the original BLAS to a compact version
        vk::CopyAccelerationStructureInfoKHR copy_info{
            *ass[i], *result[i], vk::CopyAccelerationStructureModeKHR::eCompact};
        cmd.copyAccelerationStructureKHR(copy_info);
    }

    // Make sure tlas is not build before copy finished
    vk::MemoryBarrier2 copy_tlas_barrier{vk::PipelineStageFlagBits2::eAccelerationStructureCopyKHR,
                       vk::AccessFlagBits2::eAccelerationStructureReadKHR |
                           vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                       vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                       vk::AccessFlagBits2::eAccelerationStructureReadKHR |
                           vk::AccessFlagBits2::eAccelerationStructureWriteKHR};
    vk::DependencyInfo dep_copy_tlas{{}, copy_tlas_barrier};
    cmd.pipelineBarrier2(dep_copy_tlas);

    pool->end_all();
    queue->submit(pool, fence);
    // TODO: Can this be done without waiting?
    check_result(context->device.waitForFences(fence, true, ~0), "failed waiting for fences");
    context->device.destroyFence(fence);
    context->device.destroyQueryPool(query_pool);
    return result;
}

} // namespace merian
