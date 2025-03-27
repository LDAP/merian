#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * @brief      This class describes an extension to access
 * VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME.
 *
 * Note that you must enable Ray Query or Ray Tracing Pipeline extension for this to work.
 */
class ExtensionVkAccelerationStructure : public Extension {
  public:
    ExtensionVkAccelerationStructure(
        const std::set<std::string>& required_features = {"accelerationStructure"},
        const std::set<std::string>& optional_features = {
            "accelerationStructureCaptureReplay", "accelerationStructureIndirectBuild",
            "accelerationStructureHostCommands",
            "descriptorBindingAccelerationStructureUpdateAfterBind"});

    ~ExtensionVkAccelerationStructure();

    std::vector<const char*>
    required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const override;

    // LIFECYCLE

    void on_physical_device_selected(const PhysicalDevice& pd_container) override;

    void* pnext_get_features_2(void* const p_next) override;

    bool extension_supported(const vk::Instance& /*unused*/,
                             const PhysicalDevice& /*unused*/,
                             const ExtensionContainer& /*unused*/,
                             const QueueInfo& /*unused*/) override;

    void* pnext_device_create_info(void* const p_next) override;

    const uint32_t& min_scratch_alignment() const;

  private:
    std::set<std::string> required_features;
    std::set<std::string> optional_features;

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR supported_acceleration_structure_features;
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR enabled_acceleration_structure_features;

  public:
    vk::PhysicalDeviceAccelerationStructurePropertiesKHR acceleration_structure_properties;
};

} // namespace merian
