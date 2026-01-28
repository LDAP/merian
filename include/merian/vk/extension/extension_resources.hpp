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
class ExtensionResources : public ContextExtension {
  public:
    ExtensionResources() : ContextExtension("ExtensionResources") {}
    ~ExtensionResources() {}

    std::vector<const char*>
    enable_device_extension_names(const PhysicalDeviceHandle& /*unused*/) const override;

    virtual std::vector<std::string>
    enable_device_features(const PhysicalDeviceHandle& /*unused*/) const override;

    virtual bool
    extension_supported(const std::unordered_set<std::string>& supported_instance_extensions,
                        const std::unordered_set<std::string>& supported_instance_layers) override;

    void on_physical_device_selected(const PhysicalDeviceHandle& /*unused*/) override;

    MemoryAllocatorHandle memory_allocator();
    ResourceAllocatorHandle resource_allocator();
    SamplerPoolHandle sampler_pool();
    StagingMemoryManagerHandle staging();
    DescriptorSetAllocatorHandle descriptor_pool();

  private:
    WeakContextHandle weak_context;

    // Both filled depending on device features and supported extensions.
    std::vector<const char*> required_extensions;
    VmaAllocatorCreateFlags flags{};

    std::weak_ptr<MemoryAllocator> _memory_allocator;
    std::weak_ptr<ResourceAllocator> _resource_allocator;
    std::weak_ptr<SamplerPool> _sampler_pool;
    std::weak_ptr<StagingMemoryManager> _staging;
    std::weak_ptr<DescriptorSetAllocator> _descriptor_pool;
};

} // namespace merian
