#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkRayTracingPositionFetch : public Extension {
  public:
    ExtensionVkRayTracingPositionFetch() : Extension("ExtensionVkRayTracingPositionFetch") {}
    ~ExtensionVkRayTracingPositionFetch() {}
    std::vector<const char*>
    required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const override {
        return {VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME};
    }

    void* pnext_get_features_2(void* const p_next) override {
        supported_features.setPNext(p_next);
        return &supported_features;
    }

    bool extension_supported(const vk::Instance& /*unused*/,
                             const PhysicalDevice& physical_device,
                             const ExtensionContainer& /*unused*/,
                             const QueueInfo& /*unused*/) override {
        if (physical_device.physical_device_12_properties.driverID ==
            vk::DriverId::eAmdOpenSource) {
            SPDLOG_WARN("detected AMD open-source driver. ExtensionVkRayTracingPositionFetch is "
                        "broken (last checked: 2025/05/26) - disabling!");
            return false;
        }

        return supported_features.rayTracingPositionFetch == VK_TRUE;
    }

    void* pnext_device_create_info(void* const p_next) override {
        if (supported_features.rayTracingPositionFetch == VK_TRUE) {
            SPDLOG_DEBUG("rayTracingPositionFetch supported. Enabling feature");
            enabled_features.rayTracingPositionFetch = VK_TRUE;
        } else {
            SPDLOG_ERROR("rayTracingPositionFetch requested but not supported");
        }

        enabled_features.pNext = p_next;
        return &enabled_features;
    }

  private:
    vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR supported_features;
    vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR enabled_features;
};

} // namespace merian
