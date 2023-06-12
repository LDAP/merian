#include "merian/vk/raytrace/as_compressor.hpp"

namespace merian {

// You must ensure proper synchronization with the build
std::vector<AccelerationStructureHandle>
ASCompressor::compact(const SharedContext& context,
                      const ResourceAllocatorHandle& allocator,
                      const CommandPoolHandle& pool,
                      const QueueContainerHandle& queue,
                      const std::vector<AccelerationStructureHandle>& ass,
                      const EventHandle& wait_event,
                      const vk::AccelerationStructureTypeKHR type) {
    vk::QueryPoolCreateInfo qpci{{},
                                 vk::QueryType::eAccelerationStructureCompactedSizeKHR,
                                 static_cast<uint32_t>(ass.size())};
    vk::QueryPool query_pool = context->device.createQueryPool(qpci);

    vk::CommandBuffer cmd = pool->create_and_begin();

    cmd.resetQueryPool(query_pool, 0, static_cast<uint32_t>(ass.size()));

    const vk::MemoryBarrier barrier{vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                  vk::AccessFlagBits::eAccelerationStructureWriteKHR,
                              vk::AccessFlagBits::eAccelerationStructureReadKHR |
                                  vk::AccessFlagBits::eAccelerationStructureWriteKHR};

    cmd.waitEvents(wait_event->get_event(),
                   vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                   vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, barrier, {}, {});

    // Query compacted size
    std::vector<vk::AccelerationStructureKHR> acc_structures(ass.size());
    std::transform(ass.begin(), ass.end(), acc_structures.begin(), [&](auto& as) { return as->get_acceleration_structure(); });
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

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, {}, 1, &barrier,
                        0, nullptr, 0, nullptr);

    pool->end_all();
    queue->submit(pool, fence);
    // TODO: Can this be done without waiting?
    check_result(context->device.waitForFences(fence, true, ~0), "failed waiting for fences");
    context->device.destroyFence(fence);
    context->device.destroyQueryPool(query_pool);
    return result;
}

} // namespace merian
