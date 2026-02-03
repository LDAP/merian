#pragma once

#include "merian/vk/physical_device.hpp"
#include "merian/vk/utils/vulkan_extensions.hpp"
#include "spdlog/spdlog.h"

#include <memory>

namespace merian {

class Device : public std::enable_shared_from_this<Device> {
  private:
    // features and extensions are checked for support and skipped if not available.
    Device(const PhysicalDeviceHandle& physical_device,
           const VulkanFeatures& features,
           const vk::ArrayProxyNoTemporaries<const char*>& additional_extensions,
           const vk::ArrayProxyNoTemporaries<const vk::DeviceQueueCreateInfo>& queue_create_infos,
           void* p_next)
        : physical_device(physical_device) {

        SPDLOG_DEBUG("create device");

        SPDLOG_DEBUG("...with features:");
        for (const auto& feature_name : features.get_enabled_features()) {
            if (physical_device->get_supported_features().get_feature(feature_name)) {
                SPDLOG_DEBUG("{}", feature_name);
                enabled_features.set_feature(feature_name, true);
            } else {
                SPDLOG_WARN("{} requested but not supported", feature_name);
            }
        }

        SPDLOG_DEBUG("...with extensions:");
        const uint32_t device_vk_api_version = physical_device->get_vk_api_version();
        const uint32_t instance_vk_api_version =
            physical_device->get_instance()->get_vk_api_version();

        const auto feature_extensions = enabled_features.get_required_extensions();
        std::vector<const char*> all_extensions;
        all_extensions.reserve(additional_extensions.size() + feature_extensions.size());

        const auto add_extension_recurse =
            [&](const auto& self, const ExtensionInfo* ext_info) -> std::pair<bool, std::string> {
            assert(ext_info->is_device_extension());

            // already enabled or not necessary
            if (enabled_extensions.contains(ext_info->name)) {
                return {true, ""};
            }
            if (ext_info->promoted_to_version <= device_vk_api_version) {
                SPDLOG_DEBUG("{} skipped (provided by API version)", ext_info->name);
                return {true, ""};
            }

            if (!physical_device->extension_supported(ext_info->name)) {
                return {false, fmt::format("{} not supported by physical device!", ext_info->name)};
            }

            for (const ExtensionInfo* dep : ext_info->dependencies) {
                if (dep->is_instance_extension()) {
                    // Check if promoted to instance's API version
                    if (dep->promoted_to_version <= instance_vk_api_version) {
                        continue; // Instance extension is promoted, no need to enable
                    }
                    if (!physical_device->get_instance()->extension_enabled(dep->name)) {
                        return {false,
                                fmt::format("instance extension {} is not enabled!", dep->name)};
                    }
                } else {
                    // Filter: only process if not promoted or device doesn't have promotion version
                    // yet
                    if (dep->promoted_to_version <= device_vk_api_version) {
                        continue; // Skip this dependency - it's promoted and available in device
                                  // API version
                    }
                    auto [deps_supported, reason] = self(self, dep);
                    if (!deps_supported) {
                        return {false, fmt::format("dependency {} is not supported beacause {}",
                                                   dep->name, reason)};
                    }
                }
            }

            enabled_extensions.emplace(ext_info->name);
            all_extensions.emplace_back(ext_info->name);
            SPDLOG_DEBUG(ext_info->name);
            return {true, ""};
        };

        const auto add_extension = [&](const auto* const ext) {
            if (enabled_extensions.contains(ext)) {
                return;
            }
            auto [supported, reason] =
                add_extension_recurse(add_extension_recurse, get_extension_info(ext));
            if (!supported) {
                SPDLOG_WARN("{} requested but not supported, reason: {}", ext, reason);
            }
        };

        for (const auto* const ext : additional_extensions) {
            add_extension(ext);
        }
        for (const auto* const ext : feature_extensions) {
            add_extension(ext);
        }

        void* p_next_chain = enabled_features.build_chain_for_device_creation(physical_device, p_next);
        const vk::DeviceCreateInfo device_create_info{{}, queue_create_infos, {}, all_extensions,
                                                      {}, p_next_chain};

        device = physical_device->get_physical_device().createDevice(device_create_info);
        SPDLOG_DEBUG("device ({}) created", fmt::ptr(VkDevice(device)));

        SPDLOG_DEBUG("create pipeline cache");
        vk::PipelineCacheCreateInfo pipeline_cache_create_info{};
        pipeline_cache = device.createPipelineCache(pipeline_cache_create_info);

        const vk::PhysicalDeviceFeatures& base_features = enabled_features;
        supported_pipeline_stages = vk::PipelineStageFlagBits::eVertexShader |
                                    vk::PipelineStageFlagBits::eFragmentShader |
                                    vk::PipelineStageFlagBits::eComputeShader;
        supported_pipeline_stages2 = vk::PipelineStageFlagBits2::eVertexShader |
                                     vk::PipelineStageFlagBits2::eFragmentShader |
                                     vk::PipelineStageFlagBits2::eComputeShader;
        if (base_features.tessellationShader == VK_TRUE) {
            supported_pipeline_stages |= vk::PipelineStageFlagBits::eTessellationControlShader |
                                         vk::PipelineStageFlagBits::eTessellationEvaluationShader;
            supported_pipeline_stages2 |= vk::PipelineStageFlagBits2::eTessellationControlShader |
                                          vk::PipelineStageFlagBits2::eTessellationEvaluationShader;
        }
        if (base_features.geometryShader == VK_TRUE) {
            supported_pipeline_stages |= vk::PipelineStageFlagBits::eGeometryShader;
            supported_pipeline_stages2 |= vk::PipelineStageFlagBits2::eGeometryShader;
        }
        const vk::PhysicalDeviceRayTracingPipelineFeaturesKHR& rt_pipeline_features =
            enabled_features;
        if (rt_pipeline_features.rayTracingPipeline == VK_TRUE) {
            supported_pipeline_stages |= vk::PipelineStageFlagBits::eRayTracingShaderKHR;
            supported_pipeline_stages2 |= vk::PipelineStageFlagBits2::eRayTracingShaderKHR;
        }
        const vk::PhysicalDeviceMeshShaderFeaturesEXT& mesh_shader_features = enabled_features;
        if (mesh_shader_features.meshShader == VK_TRUE) {
            supported_pipeline_stages |= vk::PipelineStageFlagBits::eMeshShaderEXT;
            supported_pipeline_stages2 |= vk::PipelineStageFlagBits2::eMeshShaderEXT;
        }
        if (mesh_shader_features.taskShader == VK_TRUE) {
            supported_pipeline_stages |= vk::PipelineStageFlagBits::eTaskShaderEXT;
            supported_pipeline_stages2 |= vk::PipelineStageFlagBits2::eTaskShaderEXT;
        }
    }

  public:
    static DeviceHandle
    create(const PhysicalDeviceHandle& physical_device,
           const VulkanFeatures& features,
           const vk::ArrayProxyNoTemporaries<const char*>& user_extensions,
           const vk::ArrayProxyNoTemporaries<const vk::DeviceQueueCreateInfo>& queue_create_infos,
           void* p_next) {
        return DeviceHandle(
            new Device(physical_device, features, user_extensions, queue_create_infos, p_next));
    }

    ~Device();

    const vk::PipelineCache& get_pipeline_cache() const {
        return pipeline_cache;
    }

    const vk::Device& get_device() const {
        return device;
    }

    const vk::Device& operator*() const {
        return device;
    }

    operator const vk::Device&() const {
        return device;
    }

    const PhysicalDeviceHandle& get_physical_device() const {
        return physical_device;
    }

    // ---------------------------------------------

    bool extension_enabled(const std::string& name) {
        return enabled_extensions.contains(name);
    }

    const std::unordered_set<std::string>& get_enabled_extensions() {
        return enabled_extensions;
    }

    const VulkanFeatures& get_enabled_features() const {
        return enabled_features;
    }

    // Shortcut for get_physical_device()->get_vk_api_version()
    // Returns the effective API version of the physical device, that is the minimum of the
    // targeted version and the supported version.
    uint32_t get_vk_api_version() const {
        return physical_device->get_vk_api_version();
    }

    // ---------------------------------------------

    vk::PipelineStageFlags get_supported_pipeline_stages() const {
        return supported_pipeline_stages;
    }

    vk::PipelineStageFlags2 get_supported_pipeline_stages2() const {
        return supported_pipeline_stages2;
    }

  private:
    const PhysicalDeviceHandle physical_device;

    std::unordered_set<std::string> enabled_extensions;
    VulkanFeatures enabled_features;

    vk::Device device;
    vk::PipelineCache pipeline_cache;

    vk::PipelineStageFlags supported_pipeline_stages;
    vk::PipelineStageFlags2 supported_pipeline_stages2;
};

using DeviceHandle = std::shared_ptr<Device>;

} // namespace merian
