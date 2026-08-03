#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * Enables VK_EXT_device_fault, if supported, to make a device loss diagnosable.
 *
 * Call dump() when a Vulkan call failed: a device loss frequently surfaces as a downstream error
 * instead of eDeviceLost, so it is fine to call it for any error.
 */
class ExtensionDeviceFault : public ContextExtension {
  public:
    static constexpr const char* name = "merian-device-fault";

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void on_device_created(const DeviceHandle& device,
                           const ExtensionContainer& extension_container) override;

    // Logs the faulting address ranges and vendor info. Logs nothing but a warning if the device
    // did not fault or the extension is unsupported.
    void dump() const;

  private:
    DeviceHandle device;
};

} // namespace merian
