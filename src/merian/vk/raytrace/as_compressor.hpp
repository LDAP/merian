#pragma once

#include "merian/vk/command/event.hpp"
#include "merian/vk/context.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

/**
 * @brief      Creates compact BLASs and TLASs.
 *
 * Compacting BLASs is recommended for static geometry to save storage space and increase
 * performance. The as compressor needs to query the compacted sizes, therefore command pool is
 * required (that is also the reason why it is not recommended to use compaction with dynamic
 * BLASs).
 *
 * Note: This is slow: The pool is submitted twice while building.
 */
class ASCompressor {
  public:
    static constexpr vk::MemoryBarrier2 build_compress_barrier{
        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
        vk::AccessFlagBits2::eAccelerationStructureReadKHR |
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
        vk::PipelineStageFlagBits2::eAccelerationStructureCopyKHR,
        vk::AccessFlagBits2::eAccelerationStructureReadKHR};

  public:
    // You must ensure proper synchronization with the build
    static std::vector<AccelerationStructureHandle>
    compact(const SharedContext& context,
            const ResourceAllocatorHandle& allocator,
            const CommandPoolHandle& pool,
            const QueueHandle& queue,
            const std::vector<AccelerationStructureHandle>& ass,
            const EventHandle& wait_event,
            const vk::AccelerationStructureTypeKHR type =
                vk::AccelerationStructureTypeKHR::eBottomLevel);
};

} // namespace merian
