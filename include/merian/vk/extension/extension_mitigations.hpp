#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * Hooks into context to prevent known driver bugs.
 */
class ExtensionMitigations : public ContextExtension {
  public:
    ExtensionMitigations() : ContextExtension("ExtensionMitigations") {}
    ~ExtensionMitigations() {}

    void on_create_device(const PhysicalDeviceHandle& physical_device,
                          vk::DeviceCreateInfo& create_info) override {
        const vk::BaseInStructure* s =
            reinterpret_cast<const vk::BaseInStructure*>(create_info.pNext);

        while (s != nullptr) {
            if (s->sType == vk::StructureType::ePhysicalDeviceRayTracingPositionFetchFeaturesKHR) {
                vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR* features =
                    const_cast<vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR*>(
                        reinterpret_cast<
                            const vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR*>(s));

                auto props =
                    physical_device->get_properties<vk::PhysicalDeviceVulkan12Properties>();

                if (props.driverID == vk::DriverId::eAmdOpenSource ||
                    props.driverID == vk::DriverId::eAmdProprietary) {
                    features->rayTracingPositionFetch = VK_FALSE;

                    SPDLOG_WARN(
                        "Mitigation: Detected AMDVLK driver. ExtensionVkRayTracingPositionFetch is "
                        "broken (last checked: 2025/07/14) - disabling!");
                }
            }

            s = s->pNext;
        }
    }
};

} // namespace merian
