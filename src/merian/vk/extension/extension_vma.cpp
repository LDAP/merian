#include "merian/vk/extension/extension_vma.hpp"
#include "merian/vk/physical_device.hpp"
#include "vk_mem_alloc.h"

#include <fmt/ranges.h>

namespace merian {

DeviceSupportInfo ExtensionVMA::query_device_support(const DeviceSupportQueryInfo& query_info) {
    DeviceSupportInfo info;
    info.supported = true; // This extension has no hard requirements

    const auto& physical_device = query_info.physical_device;

    // Add optional extensions if supported
    if (physical_device->extension_supported(VK_KHR_MAINTENANCE_4_EXTENSION_NAME)) {
        info.required_extensions.emplace_back(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
    }
    if (physical_device->extension_supported(VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) {
        info.required_extensions.emplace_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    }
    if (physical_device->extension_supported(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME)) {
        info.required_extensions.emplace_back(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME);
    }

    // Add optional features if supported
    if (physical_device->get_supported_features()
            .get_buffer_device_address_features()
            .bufferDeviceAddress == VK_TRUE) {
        info.required_features.emplace_back("bufferDeviceAddress");
    }

    return info;
}

void ExtensionVMA::on_physical_device_selected(
    const PhysicalDeviceHandle& physical_device,
    [[maybe_unused]] const ExtensionContainer& extension_container) {
    flags = {};
    if (physical_device->extension_supported(VK_KHR_MAINTENANCE_4_EXTENSION_NAME)) {
        flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
        SPDLOG_DEBUG("VMA extension: enable VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT");
    }
    if (physical_device->extension_supported(VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) {
        flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
        SPDLOG_DEBUG("VMA extension: enable VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT");
    }
    if (physical_device->extension_supported(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME)) {
        flags |= VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT;
        SPDLOG_DEBUG("VMA extension: enable VMA_ALLOCATOR_CREATE_AMD_DEVICE_COHERENT_MEMORY_BIT");
    }
    if (physical_device->get_supported_features()
            .get_buffer_device_address_features()
            .bufferDeviceAddress == VK_TRUE) {
        flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        SPDLOG_DEBUG("VMA extension: enable VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT");
    }
}

} // namespace merian
