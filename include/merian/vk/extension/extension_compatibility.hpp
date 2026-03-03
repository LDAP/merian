#pragma once

#include "merian/vk/extension/extension.hpp"

namespace merian {

/**
 * Hooks into context to prevent known driver bugs.
 */
class ExtensionCompatibility : public ContextExtension {
  private:
    static bool check_running_in_nsight(const InstanceSupportQueryInfo& query_info) {
        static constexpr std::string_view nsight_layer_prefixes[] = {
            "VK_LAYER_NV_GPU_Trace",
            "VK_LAYER_NV_ngfx_capture",
            "VK_LAYER_NV_nomad",
            "VK_LAYER_NV_shader_debugger",
        };

        for (const std::string& supported_layer : query_info.supported_layers) {
            for (const std::string_view prefix : nsight_layer_prefixes) {
                if (supported_layer.starts_with(prefix)) {
                    SPDLOG_INFO("Compatibility: Detected NVIDIA Nsight implicit layer: {}",
                                supported_layer);
                    return true;
                }
            }
        }

        return false;
    }

  public:
    void on_create_instance([[maybe_unused]] const InstanceSupportQueryInfo& support_info,
                            [[maybe_unused]] const vk::ApplicationInfo& application_info,
                            std::vector<const char*>& layer_names,
                            [[maybe_unused]] std::vector<const char*>& extension_names) override {
        // ------------------
        if (check_running_in_nsight(support_info)) {
            static constexpr std::string_view validation_layer = "VK_LAYER_KHRONOS_validation";
            const auto erased = std::erase_if(layer_names, [](const char* layer) {
                return std::string_view(layer) == validation_layer;
            });
            if (erased > 0) {
                SPDLOG_INFO("Compatibility: disabling {} to prevent conflicts", validation_layer);
            }
        }
        // ------------------
    }

    void on_create_device(const PhysicalDeviceHandle& physical_device,
                          VulkanFeatures& features,
                          std::vector<const char*>& extensions) override {

        // ------------------
        if (features.get_feature("pushDescriptor") &&
            physical_device->get_vk_api_version() < VK_API_VERSION_1_4) {
            // Feature was provided by enabling extension in earlier versions
            SPDLOG_INFO("Compatibility: pushDescriptor feature was enabled but the effective "
                        "Vulkan API Version is below 1.4. "
                        "In earlier Vulkan versions pushDescriptor was provided by "
                        "VK_KHR_push_descriptor. Enabling extension...");

            features.set_feature("pushDescriptor", false);
            extensions.emplace_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
        }
        // ------------------
    }
};

} // namespace merian
