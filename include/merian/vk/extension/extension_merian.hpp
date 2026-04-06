#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * Enables all extensions and features that are required to use merian.
 */
class ExtensionMerian : public ContextExtension {
  public:
    static constexpr const char* name = "merian";

    ExtensionMerian() : ContextExtension() {}
    ~ExtensionMerian() {}

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override {
        return DeviceSupportInfo::check(query_info,
                                        {
                                            "synchronization2", // for all kinds of sync
                                        },
                                        {
                                            "maintenance4",              // for memory allocator
                                            "samplerAnisotropy",         // for sampler pool
                                            "storageBuffer16BitAccess",              // for material system
                                            "uniformAndStorageBuffer16BitAccess",    // for material system
                                            "shaderInt16",                           // for material system
                                            "shaderFloat16",                         // for shading (half types)
                                            "accelerationStructure",                 // for scene AS
                                        },
                                        {},
                                        {
                                            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
                                            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                                        });
    }

    void on_unsupported([[maybe_unused]] const std::string& reason) override {
        throw MerianException{fmt::format("merian is unsupported on this device: {}", reason)};
    }
};

} // namespace merian
