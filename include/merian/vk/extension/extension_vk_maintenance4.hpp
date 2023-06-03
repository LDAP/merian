#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

class ExtensionVkMaintenance4 : public Extension {
  public:
    ExtensionVkMaintenance4() : Extension("ExtensionVkMaintenance4") {}
    ~ExtensionVkMaintenance4() {}

    void* pnext_get_features_2(void* const p_next) override {
        supported_features.setPNext(p_next);
        return &supported_features;
    }

    void* pnext_device_create_info(void* const p_next) override {
        if (supported_features.maintenance4) {
            SPDLOG_DEBUG("maintenance4 supported. Enabling feature");
            enable_features.maintenance4 = true;
        } else {
            SPDLOG_WARN("maintenance4 not supported");
        }

        enable_features.pNext = p_next;
        return &enable_features;
    }

  private:
    vk::PhysicalDeviceMaintenance4Features supported_features;
    vk::PhysicalDeviceMaintenance4Features enable_features;
};

} // namespace merian
