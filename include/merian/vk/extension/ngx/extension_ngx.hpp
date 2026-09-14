#pragma once

#include "merian/vk/extension/extension.hpp"

#include <string>

struct NVSDK_NGX_Parameter;

namespace merian {

// Initializes NVIDIA NGX on the device. Unsupported where merian was built without the NGX SDK or
// the device is not an NVIDIA GPU.
class ExtensionNGX : public ContextExtension {
  public:
    static constexpr const char* name = "ngx";

    ~ExtensionNGX() override;

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void on_device_created(const DeviceHandle& device,
                           const ExtensionContainer& extension_container) override;

    void on_unsupported(const std::string& reason) override;

    // Empty while NGX runs on the device.
    const std::string& get_unsupported_reason() const noexcept {
        return unsupported_reason;
    }

    // nullptr while NGX is not initialized.
    NVSDK_NGX_Parameter* get_capability_parameters() const noexcept {
        return capability_params;
    }

    const DeviceHandle& get_device() const noexcept {
        return device;
    }

  private:
    DeviceHandle device;

    std::string unsupported_reason = "NGX is not initialized";

    NVSDK_NGX_Parameter* capability_params = nullptr;
};

} // namespace merian
