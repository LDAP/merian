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

    std::shared_ptr<MemoryAllocator> memory_allocator() {
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
    std::shared_ptr<ResourceAllocator> resource_allocator() {
        if (_resource_allocator.expired()) {
            assert(!weak_context.expired());
            auto ptr = std::make_shared<ResourceAllocator>(weak_context.lock(), memory_allocator(),
                                                           staging(), sampler_pool());
            _resource_allocator = ptr;
            return ptr;
        }
        return _resource_allocator.lock();
    }
    std::shared_ptr<SamplerPool> sampler_pool() {
        if (_sampler_pool.expired()) {
            assert(!weak_context.expired());
            auto ptr = std::make_shared<SamplerPool>(weak_context.lock());
            _sampler_pool = ptr;
            return ptr;
        }
        return _sampler_pool.lock();
    }
    std::shared_ptr<StagingMemoryManager> staging() {
        if (_staging.expired()) {
            assert(!weak_context.expired());
            auto ptr =
                std::make_shared<StagingMemoryManager>(weak_context.lock(), memory_allocator());
            _staging = ptr;
            return ptr;
        }
        return _staging.lock();
    }

  private:
    std::weak_ptr<Context> weak_context;
    bool supports_device_address = false;

    std::weak_ptr<MemoryAllocator> _memory_allocator;
    std::weak_ptr<ResourceAllocator> _resource_allocator;
    std::weak_ptr<SamplerPool> _sampler_pool;
    std::weak_ptr<StagingMemoryManager> _staging;
};

} // namespace merian
