#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

/**
 * @brief      Convenience extension that initializes an memory and resource allocator.
 *
 * The extension automatically enables commonly used features (like device address) if available.
 */
class ExtensionResources : public ContextExtension {
  public:
    ExtensionResources() : ContextExtension() {}
    ~ExtensionResources() {}

    std::vector<std::string> request_extensions() override;

    void on_context_created(const ContextHandle& context,
                            const ExtensionContainer& extension_container) override;

    MemoryAllocatorHandle memory_allocator();
    ResourceAllocatorHandle resource_allocator();
    SamplerPoolHandle sampler_pool();
    StagingMemoryManagerHandle staging();
    DescriptorSetAllocatorHandle descriptor_pool();

  private:
    WeakContextHandle weak_context;

    std::weak_ptr<MemoryAllocator> _memory_allocator;
    std::weak_ptr<ResourceAllocator> _resource_allocator;
    std::weak_ptr<SamplerPool> _sampler_pool;
    std::weak_ptr<StagingMemoryManager> _staging;
    std::weak_ptr<DescriptorSetAllocator> _descriptor_pool;
};

} // namespace merian
