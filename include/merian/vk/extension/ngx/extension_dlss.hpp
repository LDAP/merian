#pragma once

#include "merian/vk/extension/extension.hpp"
#include "merian/vk/extension/ngx/dlss.hpp"
#include "merian/vk/extension/ngx/extension_ngx.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace merian {

// One NVIDIA DLSS model. Unsupported where NGX is unsupported or the device does not run the model.
class ExtensionDLSS : public ContextExtension {
  public:
    std::vector<std::string> request_extensions() override;

    InstanceSupportInfo query_instance_support(const InstanceSupportQueryInfo& query_info) override;

    DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& query_info) override;

    void on_device_created(const DeviceHandle& device,
                           const ExtensionContainer& extension_container) override;

    void on_unsupported(const std::string& reason) override;

    // Empty while the model runs on this device and driver.
    const std::string& get_unsupported_reason() const noexcept {
        return unsupported_reason;
    }

    // Render resolution DLSS asks for to reach the target resolution, and the bounds it accepts.
    // nullopt if the library does not offer the quality mode.
    std::optional<DLSSResolution> query_resolution(const vk::Extent2D& target_extent,
                                                   DLSSQuality quality) const;

    // Device memory the model holds, in bytes.
    uint64_t get_allocated_memory() const;

    // The command buffer must have completed before the first evaluate.
    DLSSHandle create(const DLSSCreateInfo& create_info, const CommandBufferHandle& cmd) const;

  protected:
    explicit ExtensionDLSS(bool ray_reconstruction);

  private:
    const bool ray_reconstruction;

    std::shared_ptr<ExtensionNGX> ngx;

    std::string unsupported_reason = "DLSS is not initialized";
};

// Antialiases and upscales.
class ExtensionDLSSSuperResolution : public ExtensionDLSS {
  public:
    static constexpr const char* name = "dlss-super-resolution";

    ExtensionDLSSSuperResolution();
};

// Denoises, antialiases and upscales a noisy input with the guides of DLSSEvalInfo.
class ExtensionDLSSRayReconstruction : public ExtensionDLSS {
  public:
    static constexpr const char* name = "dlss-ray-reconstruction";

    ExtensionDLSSRayReconstruction();
};

} // namespace merian
