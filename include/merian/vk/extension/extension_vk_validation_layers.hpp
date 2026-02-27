#pragma once

#include "merian/vk/extension/extension.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

class ExtensionVkValidationLayers : public ContextExtension {

  public:
    ExtensionVkValidationLayers() = default;

    std::vector<std::string> request_extensions() override {
        return {
            "merian-debug-utils",
        };
    }

    InstanceSupportInfo
    query_instance_support(const InstanceSupportQueryInfo& query_info) override {
        const std::string validation_layer_name = "VK_LAYER_KHRONOS_validation";

        if (!query_info.supported_layers.contains(validation_layer_name)) {
            return InstanceSupportInfo{false,
                                       fmt::format("{} not available", validation_layer_name)};
        }

        return InstanceSupportInfo{true, "", {}, {"VK_LAYER_KHRONOS_validation"}};
    }
};

} // namespace merian
