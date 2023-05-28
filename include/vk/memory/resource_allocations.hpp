#pragma once

// Possible allocations together with their memory handles.

#include "vk/memory/memory_allocator.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

struct Buffer {
    vk::Buffer buffer = VK_NULL_HANDLE;
    MemHandle memHandle{nullptr};
};

struct Image {
    vk::Image image = VK_NULL_HANDLE;
    MemHandle memHandle{nullptr};
};

struct Texture {
    vk::Image image = VK_NULL_HANDLE;
    MemHandle memHandle{nullptr};
    vk::DescriptorImageInfo descriptor{};
};

struct AccelKHR {
    vk::AccelerationStructureKHR accel = VK_NULL_HANDLE;
    Buffer buffer;
};

} // namespace merian
