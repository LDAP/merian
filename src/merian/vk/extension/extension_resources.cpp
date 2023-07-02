#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator_vma.hpp"

namespace merian {

void ExtensionResources::enable_device_features(const Context::FeaturesContainer& supported,
                                                Context::FeaturesContainer& enable) {
    if (supported.physical_device_features_v12.bufferDeviceAddress) {
        SPDLOG_DEBUG("bufferDeviceAddress supported. Enabling feature");
        enable.physical_device_features_v12.bufferDeviceAddress = true;
        supports_device_address = true;
    }
}

void ExtensionResources::on_context_created(const SharedContext context) {
    weak_context = context;
}

void ExtensionResources::on_destroy_context() {}

std::shared_ptr<MemoryAllocator> ExtensionResources::memory_allocator() {
    if (_memory_allocator.expired()) {
        assert(!weak_context.expired());
        VmaAllocatorCreateFlags flags =
            supports_device_address ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT : 0;
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
