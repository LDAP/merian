#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/memory/memory_allocator_provider.hpp"

// forward def
typedef VkFlags VmaAllocatorCreateFlags;

namespace merian {

/**
 * @brief      Load extensions and features for VulkanMemoryAllocator.
 */
class ExtensionVMA : public ContextExtension, public MemoryAllocatorProvider {
  public:
    static constexpr const char* name = "VulkanMemoryAllocator";

    ExtensionVMA() : ContextExtension() {}
    ~ExtensionVMA() {}

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void on_physical_device_selected(const PhysicalDeviceHandle& /*unused*/,
                                     const ExtensionContainer& /*extension_container*/) override;

    VmaAllocatorCreateFlags get_create_flags() const {
        return flags;
    }

    MemoryAllocatorHandle create_memory_allocator(const ContextHandle& context) const override;

  private:
    // Both filled depending on device features and supported extensions.
    std::vector<const char*> required_extensions;
    VmaAllocatorCreateFlags flags{};
};

} // namespace merian
