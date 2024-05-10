#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

/**
 * @brief      Convenience extension that initializes an memory and resource allocator.
 *
 * The extension automatically enables commonly used features (like device address) if available.
 */
class ExtensionResources : public Extension {
  public:
    ExtensionResources() : Extension("ExtensionResources") {}
    ~ExtensionResources() {}

    void enable_device_features(const Context::FeaturesContainer& supported,
                                Context::FeaturesContainer& enable) override;

    void on_context_created(const SharedContext context) override;
    void on_destroy_context() override;

    std::shared_ptr<MemoryAllocator> memory_allocator();
    std::shared_ptr<ResourceAllocator> resource_allocator();
    std::shared_ptr<SamplerPool> sampler_pool();
    std::shared_ptr<StagingMemoryManager> staging();

  private:
    std::weak_ptr<Context> weak_context;
    bool supports_device_address = false;

    std::weak_ptr<MemoryAllocator> _memory_allocator;
    std::weak_ptr<ResourceAllocator> _resource_allocator;
    std::weak_ptr<SamplerPool> _sampler_pool;
    std::weak_ptr<StagingMemoryManager> _staging;
};

} // namespace merian
