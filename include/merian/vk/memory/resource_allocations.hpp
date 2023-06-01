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

    operator vk::Buffer&() {
        return buffer;
    }

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

    // -----------------------------------------------------------

    operator vk::Image&() {
        return image;
    }
};

struct Texture {
    vk::Image image = VK_NULL_HANDLE;
    MemHandle memHandle{nullptr};
    vk::DescriptorImageInfo descriptor{};

    // -----------------------------------------------------------

    operator vk::Image&() {
        return image;
    }
};

struct AccelerationStructure {
    vk::AccelerationStructureKHR as = VK_NULL_HANDLE;
    Buffer buffer;

    // -----------------------------------------------------------

    operator Buffer&() {
        return buffer;
    }

    operator vk::Buffer&() {
        return buffer;
    }

    operator vk::AccelerationStructureKHR&() {
        return as;
    }

    // -----------------------------------------------------------

    // E.g. needed for accelerationStructureReference in VkAccelerationStructureInstanceKHR
    vk::DeviceAddress get_acceleration_structure_device_address(vk::Device& device) {
        vk::AccelerationStructureDeviceAddressInfoKHR address_info{as};
        return device.getAccelerationStructureAddressKHR(address_info);
    }
};

} // namespace merian
