#include "merian/vk/extension/extension_vk_acceleration_structure.hpp"
#include "extension_feature_macros.h"

namespace merian {

ExtensionVkAccelerationStructure::ExtensionVkAccelerationStructure(
    const std::set<std::string>& required_features, const std::set<std::string>& optional_features)
    : Extension("ExtensionVkAccelerationStructure"), required_features(required_features),
      optional_features(optional_features) {}

ExtensionVkAccelerationStructure::~ExtensionVkAccelerationStructure() {}

std::vector<const char*> ExtensionVkAccelerationStructure::required_device_extension_names(
    const vk::PhysicalDevice& /*unused*/) const {
    return {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
}

// LIFECYCLE

void ExtensionVkAccelerationStructure::on_physical_device_selected(
    const PhysicalDevice& pd_container) {
    vk::PhysicalDeviceProperties2KHR props2;
    props2.pNext = &acceleration_structure_properties;
    pd_container.physical_device.getProperties2(&props2);
}

void* ExtensionVkAccelerationStructure::pnext_get_features_2(void* const p_next) {
    supported_acceleration_structure_features.setPNext(p_next);
    return &supported_acceleration_structure_features;
}

bool ExtensionVkAccelerationStructure::extension_supported(const vk::Instance& /*unused*/,
                                                           const PhysicalDevice& /*unused*/,
                                                           const ExtensionContainer& /*unused*/,
                                                           const QueueInfo& /*unused*/) {
    bool all_required_supported = true;

    MERIAN_EXT_ENABLE_IF_REQUESTED(accelerationStructure, supported_acceleration_structure_features,
                                   enabled_acceleration_structure_features, required_features,
                                   optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(accelerationStructureCaptureReplay, supported_acceleration_structure_features,
                                   enabled_acceleration_structure_features, required_features,
                                   optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(accelerationStructureIndirectBuild, supported_acceleration_structure_features,
                                   enabled_acceleration_structure_features, required_features,
                                   optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(accelerationStructureHostCommands, supported_acceleration_structure_features,
                                   enabled_acceleration_structure_features, required_features,
                                   optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(descriptorBindingAccelerationStructureUpdateAfterBind, supported_acceleration_structure_features,
                                   enabled_acceleration_structure_features, required_features,
                                   optional_features);

    return all_required_supported;
}

void* ExtensionVkAccelerationStructure::pnext_device_create_info(void* const p_next) {
    enabled_acceleration_structure_features.pNext = p_next;
    return &enabled_acceleration_structure_features;
}

const uint32_t& ExtensionVkAccelerationStructure::min_scratch_alignment() const {
    assert(supported_acceleration_structure_features.accelerationStructure == VK_TRUE);
    return acceleration_structure_properties.minAccelerationStructureScratchOffsetAlignment;
}

} // namespace merian
