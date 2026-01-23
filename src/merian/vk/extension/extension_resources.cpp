#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_core.hpp"
#include "merian/vk/memory/memory_allocator_vma.hpp"
#include "merian/vk/physical_device.hpp"

#include <fmt/ranges.h>

namespace merian {

void ExtensionResources::on_context_initializing(const ExtensionContainer& extension_container) {
    const auto core_extension = extension_container.get_extension<ExtensionVkCore>();

    if (!core_extension) {
        SPDLOG_WARN("extension ExtensionVkCore missing. Cannot request features.");
        return;
    }

    core_extension->request_optional_feature("vk12/bufferDeviceAddress");
}

void ExtensionResources::on_physical_device_selected(const PhysicalDeviceHandle& physical_device) {
    for (const auto& extension : physical_device->physical_device_extension_properties) {
        if (strcmp(extension.extensionName, VK_KHR_MAINTENANCE_4_EXTENSION_NAME) == 0) {
            required_extensions.push_back(VK_KHR_MAINTENANCE_4_EXTENSION_NAME);
            flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
        }
        if (strcmp(extension.extensionName, VK_KHR_MAINTENANCE_5_EXTENSION_NAME) == 0) {
            required_extensions.push_back(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
            flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
        }
        if (strcmp(extension.extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
            required_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        }
    }
}

std::vector<const char*>
ExtensionResources::required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const {
    return required_extensions;
}

void ExtensionResources::on_context_created(const ContextHandle& context,
                                            const ExtensionContainer& extension_container) {
    weak_context = context;

    const auto core_extension = extension_container.get_extension<ExtensionVkCore>();

    if (core_extension) {
        if (core_extension->get_enabled_features()
                .get_physical_device_features_v12()
                .bufferDeviceAddress == VK_TRUE) {
            SPDLOG_DEBUG("bufferDeviceAddress supported. Enabling feature in allocator.");
            flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        }
    }
}

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
