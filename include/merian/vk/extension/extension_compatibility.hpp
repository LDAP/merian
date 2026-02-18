#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * Hooks into context to prevent known driver bugs.
 */
class ExtensionCompatibility : public ContextExtension {
  public:
    ExtensionCompatibility() : ContextExtension() {}
    ~ExtensionCompatibility() {}

    void on_create_device(const PhysicalDeviceHandle& physical_device,
                          VulkanFeatures& features,
                          std::vector<const char*>& extensions) override {

        // ------------------
        if (features.get_feature("pushDescriptor") &&
            physical_device->get_vk_api_version() < VK_API_VERSION_1_4) {
            // Feature was provided by enabling extension in earlier versions
            SPDLOG_INFO("Compatibility: pushDescriptor feature was enabled but the effective "
                        "Vulkan API Version is below 1.4. "
                        "In earlier Vulkan versions pushDescriptor was provided by "
                        "VK_KHR_push_descriptor. Enabling extension...");

            features.set_feature("pushDescriptor", false);
            extensions.emplace_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        }
        // ------------------
    }
};

} // namespace merian
