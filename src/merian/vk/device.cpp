#include "merian/vk/device.hpp"

#include "merian/shader/shader_defines.hpp"
#include "merian/vk/utils/vulkan_extensions.hpp"
#include "merian/vk/utils/vulkan_spirv.hpp"
#include "spdlog/spdlog.h"

namespace merian {

Device::Device(
    const PhysicalDeviceHandle& physical_device,
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
    const uint32_t instance_vk_api_version = physical_device->get_instance()->get_vk_api_version();

    const auto feature_extensions = enabled_features.get_required_extensions();
    std::vector<const char*> all_extensions;
    all_extensions.reserve(additional_extensions.size() + feature_extensions.size());

    const auto add_extension_recurse =
        [&](const auto& self, const ExtensionInfo* ext_info) -> std::pair<bool, std::string> {
        if (ext_info->is_instance_extension()) {
            // Check if promoted to instance's API version
            if (ext_info->promoted_to_version <= instance_vk_api_version) {
                return {true, ""}; // Instance extension is promoted, no need to enable
            }
            if (physical_device->get_instance()->extension_enabled(ext_info->name)) {
                return {true, ""};
            } else {
                return {false,
                        fmt::format("instance extension {} is not enabled!", ext_info->name)};
            }
        }

        // already enabled or not necessary
        if (enabled_extensions.contains(ext_info->name)) {
            return {true, ""};
        }
        if (ext_info->promoted_to_version <= device_vk_api_version) {
            SPDLOG_DEBUG("{} skipped (provided by API version)", ext_info->name);
            return {true, ""};
        }
        if (ext_info->deprecated_by != nullptr &&
            physical_device->extension_supported(ext_info->deprecated_by->name)) {
            SPDLOG_DEBUG("{} skipped (deprecated by {})", ext_info->name,
                         ext_info->deprecated_by->name);
            return self(self, ext_info->deprecated_by);
        }

        if (!physical_device->extension_supported(ext_info->name)) {
            return {false, fmt::format("{} not supported by physical device!", ext_info->name)};
        }

        for (const ExtensionInfo* dep : ext_info->dependencies) {
            auto [deps_supported, reason] = self(self, dep);
            if (!deps_supported) {
                return {false, fmt::format("dependency {} is not supported beacause {}", dep->name,
                                           reason)};
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
    const vk::PhysicalDeviceRayTracingPipelineFeaturesKHR& rt_pipeline_features = enabled_features;
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

    for (const auto& ext : get_spirv_extensions()) {
        const auto device_extension_deps =
            get_spirv_extension_requirements(ext, physical_device->get_vk_api_version());
        bool all_enabled = true;
        for (const auto& dep : device_extension_deps) {
            all_enabled &= enabled_extensions.contains(dep);
        }
        if (all_enabled) {
            enabled_spirv_extensions.insert(ext);
        }
    }

    for (const auto& cap : get_spirv_capabilities()) {
        if (is_spirv_capability_supported(cap, physical_device->get_vk_api_version(),
                                          enabled_features, physical_device->get_properties())) {
            enabled_spirv_capabilities.insert(cap);
        }
    }

    // Precompute shader defines
    for (const auto& ext : physical_device->get_instance()->get_enabled_extensions()) {
        shader_defines.emplace(std::string(SHADER_DEFINE_PREFIX_INSTANCE_EXT) + ext, "1");
    }

    for (const auto& ext : enabled_extensions) {
        shader_defines.emplace(std::string(SHADER_DEFINE_PREFIX_DEVICE_EXT) + ext, "1");
    }

    for (const auto& ext : enabled_spirv_extensions) {
        shader_defines.emplace(std::string(SHADER_DEFINE_PREFIX_SPIRV_EXT) + ext, "1");
    }

    for (const auto& cap : enabled_spirv_capabilities) {
        shader_defines.emplace(std::string(SHADER_DEFINE_PREFIX_SPIRV_CAP) + cap, "1");
    }

    vk_get_device_proc_addr =
        PFN_vkGetDeviceProcAddr(vkGetDeviceProcAddr(device, "vkGetDeviceProcAddr"));
}

Device::~Device() {
    device.waitIdle();

    SPDLOG_DEBUG("destroy pipeline cache");
    device.destroyPipelineCache(pipeline_cache);

    SPDLOG_DEBUG("destroy device");
    device.destroy();
}

const std::unordered_set<std::string>& Device::get_enabled_spirv_extensions() const {
    return enabled_spirv_extensions;
}

const std::unordered_set<std::string>& Device::get_enabled_spirv_capabilities() const {
    return enabled_spirv_capabilities;
}

const std::map<std::string, std::string>& Device::get_shader_defines() const {
    return shader_defines;
}

} // namespace merian
