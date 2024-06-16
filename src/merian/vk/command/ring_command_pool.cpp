#include "ring_command_pool.hpp"

namespace merian {

RingCommandPoolCycle::RingCommandPoolCycle(const SharedContext& context,
                                           const uint32_t queue_family_index,
                                           const vk::CommandPoolCreateFlags create_flags,
                                           const uint32_t cycle_index,
                                           const uint32_t& current_index)
    : CommandPool(context, queue_family_index, create_flags), cycle_index(cycle_index),
      current_index(current_index){};

vk::CommandBuffer
RingCommandPoolCycle::create(const vk::CommandBufferLevel level,
                             const bool begin,
                             const vk::CommandBufferUsageFlags flags,
                             const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    assert(current_index == cycle_index && "do not use pools from an other cycle");
    return CommandPool::create(level, begin, flags, pInheritanceInfo);
}

vk::CommandBuffer
RingCommandPoolCycle::create_and_begin(const vk::CommandBufferLevel level,
                                       const vk::CommandBufferUsageFlags flags,
                                       const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    assert(current_index == cycle_index && "do not use pools from an other cycle");
    return CommandPool::create_and_begin(level, flags, pInheritanceInfo);
}

std::vector<vk::CommandBuffer>
RingCommandPoolCycle::create_multiple(const vk::CommandBufferLevel level,
                                      const uint32_t count,
                                      const bool begin,
                                      const vk::CommandBufferUsageFlags flags,
                                      const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    assert(current_index == cycle_index && "do not use pools from an other cycle");
    return CommandPool::create_multiple(level, count, begin, flags, pInheritanceInfo);
}

std::vector<vk::CommandBuffer> RingCommandPoolCycle::create_and_begin_multiple(
    const vk::CommandBufferLevel level,
    const uint32_t count,
    const vk::CommandBufferUsageFlags flags,
    const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    assert(current_index == cycle_index && "do not use pools from an other cycle");
    return CommandPool::create_and_begin_multiple(level, count, flags, pInheritanceInfo);
}

} // namespace merian
