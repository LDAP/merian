#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator_vma.hpp"
#include "merian/vk/physical_device.hpp"

#include <fmt/ranges.h>

namespace merian {

DeviceSupportInfo
ExtensionResources::query_device_support(const DeviceSupportQueryInfo& query_info) {
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

void ExtensionResources::on_physical_device_selected(const PhysicalDeviceHandle& physical_device) {
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

void ExtensionResources::on_context_created(const ContextHandle& context,
                                            const ExtensionContainer& /*extension_container*/) {
    weak_context = context;
}

// --------------------

MemoryAllocatorHandle ExtensionResources::memory_allocator() {
    if (_memory_allocator.expired()) {
        assert(!weak_context.expired());
        auto ptr = VMAMemoryAllocator::create(weak_context.lock(), flags);
        _memory_allocator = ptr;
        return ptr;
    }
    return _memory_allocator.lock();
}
ResourceAllocatorHandle ExtensionResources::resource_allocator() {
    if (_resource_allocator.expired()) {
        assert(!weak_context.expired());
        auto ptr = std::make_shared<ResourceAllocator>(
            weak_context.lock(), memory_allocator(), staging(), sampler_pool(), descriptor_pool());
        _resource_allocator = ptr;
        return ptr;
    }
    return _resource_allocator.lock();
}
SamplerPoolHandle ExtensionResources::sampler_pool() {
    if (_sampler_pool.expired()) {
        assert(!weak_context.expired());
        auto ptr = std::make_shared<SamplerPool>(weak_context.lock());
        _sampler_pool = ptr;
        return ptr;
    }
    return _sampler_pool.lock();
}
StagingMemoryManagerHandle ExtensionResources::staging() {
    if (_staging.expired()) {
        assert(!weak_context.expired());
        auto ptr = std::make_shared<StagingMemoryManager>(memory_allocator());
        _staging = ptr;
        return ptr;
    }
    return _staging.lock();
}
DescriptorSetAllocatorHandle ExtensionResources::descriptor_pool() {
    if (_descriptor_pool.expired()) {
        assert(!weak_context.expired());
        auto ptr = ResizingDescriptorPool::create(weak_context.lock());
        _descriptor_pool = ptr;
        return ptr;
    }
    return _descriptor_pool.lock();
}

} // namespace merian
