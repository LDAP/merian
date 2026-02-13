#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * Hooks into context to prevent known driver bugs.
 */
class ExtensionMitigations : public ContextExtension {
  public:
    ExtensionMitigations() : ContextExtension() {}
    ~ExtensionMitigations() {}

    void on_create_device(const PhysicalDeviceHandle& physical_device,
                          VulkanFeatures& features,
                          std::vector<const char*>& extensions) override {

        // ------------------
        if (physical_device->get_properties().is_available<vk::PhysicalDeviceDriverProperties>()) {
            // Mitigation: ExtensionVkRayTracingPositionFetch on AMDVLK
            const vk::PhysicalDeviceDriverProperties props = physical_device->get_properties();
            if ((props.driverID == vk::DriverId::eAmdOpenSource ||
                 props.driverID == vk::DriverId::eAmdProprietary) &&
                features.get_ray_tracing_position_fetch_features_khr().rayTracingPositionFetch ==
                    VK_TRUE) {

                SPDLOG_WARN(
                    "Mitigation: Detected AMDVLK driver. ExtensionVkRayTracingPositionFetch is "
                    "broken (last checked: 2025/07/14) - disabling!");

                features.set_feature("rayTracingPositionFetch", false);
                auto it = extensions.begin();
                while (it != extensions.end()) {
                    if (strcmp(*it, VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME) == 0) {
                        it = extensions.erase(it);
                    } else {
                        it++;
                    }
                }
            }
        }
        // ------------------
    }
};

} // namespace merian
