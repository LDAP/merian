#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

enum ExtensionVkVulkanMemoryModelFlags : uint8_t {
    EXTENSION_VK_VULKAN_MEMORY_MODEL_FLAGS_DEVICE_SCOPE = 0x1,
    EXTENSION_VK_VULKAN_MEMORY_MODEL_FLAGS_AVAILABILITY_VISIBILITY_CHAINS = 0x2,
};

class ExtensionVkVulkanMemoryModel : public Extension {
  public:
    ExtensionVkVulkanMemoryModel(const ExtensionVkVulkanMemoryModelFlags flags)
        : Extension("ExtensionVkVulkanMemoryModel"), flags(flags) {}
    ~ExtensionVkVulkanMemoryModel() override = default;

    void enable_device_features(const Context::FeaturesContainer& supported_features,
                                        Context::FeaturesContainer& enabled_features) override {
        if (supported_features.physical_device_features_v12.vulkanMemoryModel == VK_TRUE) {
            SPDLOG_DEBUG("vulkanMemoryModel supported. Enabling feature");
            enabled_features.physical_device_features_v12.vulkanMemoryModel = VK_TRUE;
        } else {
            SPDLOG_WARN("vulkanMemoryModel not supported");
        }
        if ((flags & EXTENSION_VK_VULKAN_MEMORY_MODEL_FLAGS_DEVICE_SCOPE) != 0) {
            if (supported_features.physical_device_features_v12.vulkanMemoryModelDeviceScope == VK_TRUE) {
                SPDLOG_DEBUG("vulkanMemoryModelDeviceScope supported. Enabling feature");
                enabled_features.physical_device_features_v12.vulkanMemoryModelDeviceScope = VK_TRUE;
            } else {
                SPDLOG_WARN("vulkanMemoryModelDeviceScope not supported");
            }
        }
        if ((flags & EXTENSION_VK_VULKAN_MEMORY_MODEL_FLAGS_AVAILABILITY_VISIBILITY_CHAINS) != 0) {
            if (supported_features.physical_device_features_v12.vulkanMemoryModelAvailabilityVisibilityChains == VK_TRUE) {
                SPDLOG_DEBUG(
                    "vulkanMemoryModelAvailabilityVisibilityChains supported. Enabling feature");
                enabled_features.physical_device_features_v12.vulkanMemoryModelAvailabilityVisibilityChains = VK_TRUE;
            } else {
                SPDLOG_WARN("vulkanMemoryModelAvailabilityVisibilityChains not supported");
            }
        }
    }

  private:
    const ExtensionVkVulkanMemoryModelFlags flags;
};

} // namespace merian
