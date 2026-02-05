#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/extension/extension.hpp"

#include <vulkan/vulkan.hpp>

namespace merian {

class ExtensionVkLayerSettings : public ContextExtension {
  public:
    inline static const vk::LayerSettingEXT ENABLE_VALIDATION_LAYER_PRINTF{
        "VK_LAYER_KHRONOS_validation",
        "enables",
        vk::LayerSettingTypeEXT::eString,
        1,
        &"VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT",
    };

  public:
    ExtensionVkLayerSettings(const vk::ArrayProxy<vk::LayerSettingEXT>& settings)
        : ContextExtension("ExtensionVkLayerSettings"), settings(settings.begin(), settings.end()) {}

    // Overrides
    ~ExtensionVkLayerSettings() {}
    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& /*query_info*/) override {
        InstanceSupportInfo info;
        info.required_extensions = {
            VK_EXT_LAYER_SETTINGS_EXTENSION_NAME,
        };
        return info;
    }

    void* pnext_instance_create_info(void* const p_next) override {
        layer_settings_create_info.settingCount = settings.size();
        layer_settings_create_info.setSettings(settings);
        layer_settings_create_info.setPNext(p_next);

        return &layer_settings_create_info;
    }

  private:
    const std::vector<vk::LayerSettingEXT> settings;
    vk::LayerSettingsCreateInfoEXT layer_settings_create_info;
};

} // namespace merian
