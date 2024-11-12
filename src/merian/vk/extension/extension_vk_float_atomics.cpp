#include "merian/vk/extension/extension_vk_float_atomics.hpp"

#include "extension_feature_macros.h"

namespace merian {

ExtensionVkFloatAtomics::ExtensionVkFloatAtomics(const std::set<std::string>& required_features,
                                                 const std::set<std::string>& optional_features)
    : Extension("ExtensionVkFloatAtomics"), required_features(required_features),
      optional_features(optional_features) {}

ExtensionVkFloatAtomics::~ExtensionVkFloatAtomics() {}

std::vector<const char*> ExtensionVkFloatAtomics::required_device_extension_names(
    const vk::PhysicalDevice& /*unused*/) const {
    return {
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
    };
}

void* ExtensionVkFloatAtomics::pnext_get_features_2(void* const p_next) {
    supported_atomic_features.setPNext(p_next);
    return &supported_atomic_features;
}

bool ExtensionVkFloatAtomics::extension_supported(const vk::Instance& /*unused*/,
                                                  const PhysicalDevice& /*unused*/,
                                                  const ExtensionContainer& /*unused*/,
                                                  const QueueInfo& /*unused*/) {
    bool all_required_supported = true;

    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderBufferFloat32Atomics, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderBufferFloat32AtomicAdd, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderBufferFloat64Atomics, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderBufferFloat64AtomicAdd, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderSharedFloat32Atomics, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderSharedFloat32AtomicAdd, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderSharedFloat64Atomics, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderSharedFloat64AtomicAdd, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderImageFloat32Atomics, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(shaderImageFloat32AtomicAdd, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(sparseImageFloat32Atomics, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);
    MERIAN_EXT_ENABLE_IF_REQUESTED(sparseImageFloat32AtomicAdd, supported_atomic_features,
                                   enabled_atomic_features, required_features, optional_features);

    for (const auto& unknown_feature : optional_features) {
        throw std::invalid_argument{fmt::format("feature {} is unknown", unknown_feature)};
    }
    for (const auto& unknown_feature : required_features) {
        throw std::invalid_argument{fmt::format("feature {} is unknown", unknown_feature)};
    }

    return all_required_supported;
}

void* ExtensionVkFloatAtomics::pnext_device_create_info(void* const p_next) {
    enabled_atomic_features.pNext = p_next;
    return &enabled_atomic_features;
}

// --------------------------------------------------------------------------

const vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT&
ExtensionVkFloatAtomics::get_supported_features() const {
    return supported_atomic_features;
}

const vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT&
ExtensionVkFloatAtomics::get_enabled_features() const {
    return enabled_atomic_features;
}

} // namespace merian
