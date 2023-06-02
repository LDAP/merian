#pragma once

#include "merian/vk/extension/extension.hpp"

#include "merian/vk/memory/resource_allocator.hpp"
#include "vk_mem_alloc.h"

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
                                        Context::FeaturesContainer& enable) override {
        if (supported.physical_device_features_v12.bufferDeviceAddress) {
            SPDLOG_DEBUG("bufferDeviceAddress supported. Enabling feature");
            enable.physical_device_features_v12.bufferDeviceAddress = true;
            supports_device_address = true;
        }
        if (supported.physical_device_features_v12.uniformAndStorageBuffer8BitAccess) {
            SPDLOG_DEBUG("uniformAndStorageBuffer8BitAccess supported. Enabling feature");
            enable.physical_device_features_v12.uniformAndStorageBuffer8BitAccess = true;
        }
        if (supported.physical_device_features_v12.descriptorIndexing) {
            SPDLOG_DEBUG("descriptorIndexing supported. Enabling feature");
            enable.physical_device_features_v12.descriptorIndexing = true;
        }
        if (supported.physical_device_features_v12.shaderSampledImageArrayNonUniformIndexing) {
            SPDLOG_DEBUG("shaderSampledImageArrayNonUniformIndexing supported. Enabling feature");
            enable.physical_device_features_v12.shaderSampledImageArrayNonUniformIndexing = true;
        }
        if (supported.physical_device_features_v12.runtimeDescriptorArray) {
            SPDLOG_DEBUG("runtimeDescriptorArray supported. Enabling feature");
            enable.physical_device_features_v12.runtimeDescriptorArray = true;
        }
    }

    void on_context_created(const SharedContext context) override;
    void on_destroy_context() override;

    MemoryAllocator& memory_allocator() {
        return *_memory_allocator;
    }
    ResourceAllocator& resource_allocator() {
        return *_resource_allocator;
    }
    SamplerPool& sampler_pool() {
        return *_sampler_pool;
    }

  private:
    bool supports_device_address = false;
    VmaAllocator vma_allocator = VK_NULL_HANDLE;
    MemoryAllocator* _memory_allocator;
    ResourceAllocator* _resource_allocator;
    SamplerPool* _sampler_pool;
};

} // namespace merian
