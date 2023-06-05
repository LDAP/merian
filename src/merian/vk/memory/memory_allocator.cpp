#include "merian/vk/memory/memory_allocator.hpp"

#include <cassert>

namespace merian {

uint32_t getMemoryType(const vk::PhysicalDeviceMemoryProperties& memoryProperties, uint32_t typeBits,
                       const vk::MemoryPropertyFlags& properties) {
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if (((typeBits & (1 << i)) > 0) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    assert(0);
    return ~0u;
}

} // namespace merian
