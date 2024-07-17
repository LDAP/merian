#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/memory/memory_allocator_vma.hpp"
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

    void on_physical_device_selected(const Context::PhysicalDeviceContainer&) override;

    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override;

    void enable_device_features(const Context::FeaturesContainer& supported,
                                Context::FeaturesContainer& enable) override;

    void on_context_created(const ContextHandle context) override;
    void on_destroy_context() override;

    std::shared_ptr<MemoryAllocator> memory_allocator();
    std::shared_ptr<ResourceAllocator> resource_allocator();
    std::shared_ptr<SamplerPool> sampler_pool();
    std::shared_ptr<StagingMemoryManager> staging();

  private:
    std::weak_ptr<Context> weak_context;

    // Both filled depending on device features and supported extensions.
    std::vector<const char*> required_extensions;
    VmaAllocatorCreateFlags flags{};

    std::weak_ptr<MemoryAllocator> _memory_allocator;
    std::weak_ptr<ResourceAllocator> _resource_allocator;
    std::weak_ptr<SamplerPool> _sampler_pool;
    std::weak_ptr<StagingMemoryManager> _staging;
};

} // namespace merian
