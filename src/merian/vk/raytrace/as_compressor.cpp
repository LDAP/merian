#include "merian/vk/raytrace/as_compressor.hpp"
#include "merian/vk/command/command_buffer.hpp"
#include "merian/vk/utils/check_result.hpp"

namespace merian {

// You must ensure proper synchronization with the build
std::vector<AccelerationStructureHandle>
ASCompressor::compact(const ContextHandle& context,
                      const ResourceAllocatorHandle& allocator,
                      const CommandPoolHandle& pool,
                      const QueueHandle& queue,
                      const std::vector<AccelerationStructureHandle>& ass,
                      const EventHandle& build_wait_event,
                      const vk::AccelerationStructureTypeKHR type) {
    // vk::QueryPoolCreateInfo qpci{{},
    //                              vk::QueryType::eAccelerationStructureCompactedSizeKHR,
    //                              static_cast<uint32_t>(ass.size())};
    // vk::QueryPool query_pool = context->device.createQueryPool(qpci);

    QueryPoolHandle<vk::QueryType::eAccelerationStructureCompactedSizeKHR> query_pool =
        QueryPool<vk::QueryType::eAccelerationStructureCompactedSizeKHR>::create(context,
                                                                                 ass.size());

    const CommandBufferHandle cmd = CommandBuffer::create(pool);
    cmd->begin();
    cmd->reset(query_pool);

    // Wait for build to complete
    cmd->set_event(build_wait_event, ASCompressor::build_compress_barrier);

    // Query compacted size
    cmd->write_acceleration_structures_properties(query_pool, ass);
    vk::Fence fence = context->device.createFence({});
    cmd->end();
    queue->submit(cmd, fence);

    // TODO: Can this be done without waiting?
    check_result(
        context->device.waitForFences(fence, VK_TRUE, std::numeric_limits<uint64_t>::max()),
        "failed waiting for fences");
    pool->reset();
    context->device.resetFences(fence);
    cmd->begin();

    std::vector<vk::DeviceSize> compact_sizes =
        query_pool->get_query_pool_results<vk::DeviceSize>(vk::QueryResultFlagBits::eWait);

    std::vector<AccelerationStructureHandle> result(ass.size());
    for (uint32_t i = 0; i < ass.size(); i++) {

        auto size_info = ass[i]->get_size_info();
        size_info.accelerationStructureSize = compact_sizes[i];

        // Creating a compact version of the AS
        result[i] = allocator->create_acceleration_structure(type, size_info);

        // Copy the original BLAS to a compact version
        cmd->copy_acceleration_structure(ass[i], result[i],
                                         vk::CopyAccelerationStructureModeKHR::eCompact);
    }

    // Make sure tlas is not build before copy finished
    const vk::MemoryBarrier2 copy_tlas_barrier{
        vk::PipelineStageFlagBits2::eAccelerationStructureCopyKHR,
        vk::AccessFlagBits2::eAccelerationStructureReadKHR |
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        vk::AccessFlagBits2::eAccelerationStructureReadKHR |
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR};
    cmd->barrier(copy_tlas_barrier);

    cmd->end();
    queue->submit(cmd, fence);

    // TODO: Can this be done without waiting?
    check_result(
        context->device.waitForFences(fence, VK_TRUE, std::numeric_limits<uint64_t>::max()),
        "failed waiting for fences");
    context->device.destroyFence(fence);
    return result;
}

} // namespace merian
