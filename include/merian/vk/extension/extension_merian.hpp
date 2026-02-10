#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * Enables all extensions and features that are required to use merian.
 */
class ExtensionMerian : public ContextExtension {
  public:
    ExtensionMerian() : ContextExtension() {}
    ~ExtensionMerian() {}

    std::vector<std::string> request_extensions() override {
        return {
            "merian-mitigations",
        };
    }

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override {
        return DeviceSupportInfo::check(query_info,
                                        {
                                            "synchronization2", // for all kinds of sync
                                        },
                                        {
                                            "maintenance4",     // for memory allocator
                                            "samplerAnisotropy" // for sampler pool

                                        },
                                        {},
                                        {
                                            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
                                        });
    }

    void on_unsupported([[maybe_unused]] const std::string& reason) override {
        throw MerianException{fmt::format("merian is unsupported on this device: {}", reason)};
    }
};

} // namespace merian
