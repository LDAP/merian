#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkFloatAtomics : public Extension {
  public:
    ExtensionVkFloatAtomics(const std::set<std::string>& required_features = {},
                            const std::set<std::string>& optional_features = {
                                "shaderBufferFloat32Atomics", "shaderBufferFloat32AtomicAdd",
                                "shaderBufferFloat64Atomics", "shaderBufferFloat64AtomicAdd",
                                "shaderSharedFloat32Atomics", "shaderSharedFloat32AtomicAdd",
                                "shaderSharedFloat64Atomics", "shaderSharedFloat64AtomicAdd",
                                "shaderImageFloat32Atomics", "shaderImageFloat32AtomicAdd",
                                "sparseImageFloat32Atomics", "sparseImageFloat32AtomicAdd"});
    ~ExtensionVkFloatAtomics();

    std::vector<const char*>
    required_device_extension_names(const vk::PhysicalDevice& /*unused*/) const override;

    void* pnext_get_features_2(void* const p_next) override;

    bool extension_supported(const vk::Instance& /*unused*/,
                             const PhysicalDevice& /*unused*/,
                             const ExtensionContainer& /*unused*/,
                             const QueueInfo& /*unused*/) override;

    void* pnext_device_create_info(void* const p_next) override;

    // --------------------------------------------------------------------------

    const vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT& get_supported_features() const;

    const vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT& get_enabled_features() const;

  private:
    std::set<std::string> required_features;
    std::set<std::string> optional_features;

    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT supported_atomic_features;
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT enabled_atomic_features;
};

} // namespace merian
