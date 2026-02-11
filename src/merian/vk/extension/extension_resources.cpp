#include "merian/vk/extension/extension_resources.hpp"
#include "merian/vk/memory/memory_allocator_vma.hpp"

#include <fmt/ranges.h>

namespace merian {

std::vector<std::string> ExtensionResources::request_extensions() {
    return {"merian-vma"};
}

void ExtensionResources::on_context_created(const ContextHandle& context,
                                            const ExtensionContainer& /*extension_container*/) {
    weak_context = context;
}

// --------------------

MemoryAllocatorHandle ExtensionResources::memory_allocator() {
    if (_memory_allocator.expired()) {
        assert(!weak_context.expired());
        auto ptr = VMAMemoryAllocator::create(weak_context.lock());
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
