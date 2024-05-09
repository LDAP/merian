#include "ring_command_pool.hpp"

namespace merian {

RingCommandPoolCycle::RingCommandPoolCycle(const SharedContext& context,
                                           uint32_t queue_family_index,
                                           vk::CommandPoolCreateFlags create_flags,
                                           uint32_t cycle_index,
                                           uint32_t& current_index)
    : CommandPool(context, queue_family_index, create_flags), cycle_index(cycle_index),
      current_index(current_index){};

vk::CommandBuffer
RingCommandPoolCycle::create(vk::CommandBufferLevel level,
                             bool begin,
                             vk::CommandBufferUsageFlags flags,
                             const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    assert(current_index == cycle_index && "do not use pools from an other cycle");
    return CommandPool::create(level, begin, flags, pInheritanceInfo);
}

vk::CommandBuffer
RingCommandPoolCycle::create_and_begin(vk::CommandBufferLevel level,
                                       vk::CommandBufferUsageFlags flags,
                                       const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    assert(current_index == cycle_index && "do not use pools from an other cycle");
    return CommandPool::create_and_begin(level, flags, pInheritanceInfo);
}

std::vector<vk::CommandBuffer>
RingCommandPoolCycle::create_multiple(vk::CommandBufferLevel level,
                                      uint32_t count,
                                      bool begin,
                                      vk::CommandBufferUsageFlags flags,
                                      const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    assert(current_index == cycle_index && "do not use pools from an other cycle");
    return CommandPool::create_multiple(level, count, begin, flags, pInheritanceInfo);
}

std::vector<vk::CommandBuffer> RingCommandPoolCycle::create_and_begin_multiple(
    vk::CommandBufferLevel level,
    uint32_t count,
    vk::CommandBufferUsageFlags flags,
    const vk::CommandBufferInheritanceInfo* pInheritanceInfo) {
    assert(current_index == cycle_index && "do not use pools from an other cycle");
    return CommandPool::create_and_begin_multiple(level, count, flags, pInheritanceInfo);
}

} // namespace merian
