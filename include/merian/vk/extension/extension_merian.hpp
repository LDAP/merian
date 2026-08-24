#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/pipeline/pipeline.hpp"

#include <vector>

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
        std::vector<const char*> optional_features{
            "maintenance4",      // for memory allocator
            "samplerAnisotropy", // for sampler pool
            "scalarBlockLayout", // for scalar-layout buffers
        };
        std::vector<const char*> optional_extensions{
            VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        };
        // The driver keeps every pipeline's compiler statistics alive while this is enabled, which
        // it charges to pipeline creation, so it stays off unless the statistics are asked for.
        if (shader_stats_requested()) {
            optional_features.emplace_back("pipelineExecutableInfo");
            optional_extensions.emplace_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        }

        return DeviceSupportInfo::check(query_info,
                                        {
                                            "synchronization2", // for all kinds of sync
                                        },
                                        optional_features, {}, optional_extensions);
    }

    void on_unsupported([[maybe_unused]] const std::string& reason) override {
        throw MerianException{fmt::format("merian is unsupported on this device: {}", reason)};
    }
};

} // namespace merian
