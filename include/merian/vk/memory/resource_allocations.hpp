#pragma once

// Possible allocations together with their memory handles.

#include "merian/vk/memory/memory_allocator.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

struct Buffer {
    vk::Buffer buffer = VK_NULL_HANDLE;
    MemHandle memHandle{nullptr};
    vk::BufferUsageFlags usage;

    // -----------------------------------------------------------

    vk::BufferDeviceAddressInfo get_buffer_device_address_info() {
        return vk::BufferDeviceAddressInfo{buffer};
    }

    vk::DeviceAddress get_device_address(vk::Device& device) {
        assert(usage | vk::BufferUsageFlagBits::eShaderDeviceAddress);
        return device.getBufferAddress(get_buffer_device_address_info());
    }
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

struct AccelerationStructure {
    vk::AccelerationStructureKHR as = VK_NULL_HANDLE;
    Buffer buffer;

    // -----------------------------------------------------------

    vk::DeviceAddress get_acceleration_structure_device_address(vk::Device& device) {
        vk::AccelerationStructureDeviceAddressInfoKHR address_info{as};
        return device.getAccelerationStructureAddressKHR(address_info);
    }
};

} // namespace merian
