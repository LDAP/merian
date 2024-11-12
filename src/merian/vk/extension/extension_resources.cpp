#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/extension/extension_vk_core.hpp"
#include "merian/vk/memory/memory_allocator_vma.hpp"

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

void ExtensionResources::on_physical_device_selected(const PhysicalDevice& physical_device) {
    for (const auto& extension : physical_device.physical_device_extension_properties) {
        if (strcmp(extension.extensionName, "VK_KHR_maintenance4") == 0) {
            required_extensions.push_back("VK_KHR_maintenance4");
            flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE4_BIT;
        }
        if (strcmp(extension.extensionName, "VK_KHR_maintenance5") == 0) {
            required_extensions.push_back("VK_KHR_maintenance5");
            flags |= VMA_ALLOCATOR_CREATE_KHR_MAINTENANCE5_BIT;
        }
        if (strcmp(extension.extensionName, "VK_KHR_buffer_device_address") == 0) {
            required_extensions.push_back("VK_KHR_buffer_device_address");
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

void ExtensionResources::on_destroy_context() {}

std::shared_ptr<MemoryAllocator> ExtensionResources::memory_allocator() {
    if (_memory_allocator.expired()) {
        assert(!weak_context.expired());
        auto ptr = VMAMemoryAllocator::make_allocator(weak_context.lock(), flags);
        _memory_allocator = ptr;
        return ptr;
    }
    return _memory_allocator.lock();
}
std::shared_ptr<ResourceAllocator> ExtensionResources::resource_allocator() {
    if (_resource_allocator.expired()) {
        assert(!weak_context.expired());
        auto ptr = std::make_shared<ResourceAllocator>(weak_context.lock(), memory_allocator(),
                                                       staging(), sampler_pool());
        _resource_allocator = ptr;
        return ptr;
    }
    return _resource_allocator.lock();
}
std::shared_ptr<SamplerPool> ExtensionResources::sampler_pool() {
    if (_sampler_pool.expired()) {
        assert(!weak_context.expired());
        auto ptr = std::make_shared<SamplerPool>(weak_context.lock());
        _sampler_pool = ptr;
        return ptr;
    }
    return _sampler_pool.lock();
}
std::shared_ptr<StagingMemoryManager> ExtensionResources::staging() {
    if (_staging.expired()) {
        assert(!weak_context.expired());
        auto ptr = std::make_shared<StagingMemoryManager>(weak_context.lock(), memory_allocator());
        _staging = ptr;
        return ptr;
    }
    return _staging.lock();
}

} // namespace merian
