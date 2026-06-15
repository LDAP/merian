#include "merian/vk/device.hpp"

#include "fmt/ranges.h"
#include "merian/shader/shader_defines.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/utils/vulkan_extensions.hpp"
#include "merian/vk/utils/vulkan_spirv.hpp"
#include "spdlog/spdlog.h"

namespace merian {

Device::Device(
    const PhysicalDeviceHandle& physical_device,
    const VulkanFeatures& features,
    const vk::ArrayProxyNoTemporaries<const char*>& extensions,
    const vk::ArrayProxyNoTemporaries<const vk::DeviceQueueCreateInfo>& queue_create_infos,
    void* p_next)
    : physical_device(physical_device), enabled_extensions(extensions.begin(), extensions.end()),
      enabled_features(features) {

    void* p_next_chain = enabled_features.build_chain_for_device_creation(physical_device, p_next);
    const vk::DeviceCreateInfo device_create_info{
        {}, queue_create_infos, {}, extensions, {}, p_next_chain,
    };

    device = physical_device->get_physical_device().createDevice(device_create_info);
    if (spdlog::should_log(spdlog::level::debug)) {
        SPDLOG_INFO(
            "device ({}) created. (Vulkan supported: {}, target: {}, effective: "
            "{}, features: [{}], extensions: [{}])",
            fmt::ptr(VkDevice(device)),
            format_vk_api_version(physical_device->get_physical_device_vk_api_version()),
            format_vk_api_version(physical_device->get_instance()->get_target_vk_api_version()),
            format_vk_api_version(physical_device->get_vk_api_version()),
            fmt::join(features.get_enabled_features(), ", "), fmt::join(enabled_extensions, ", "));
    } else {
        SPDLOG_INFO("device ({}) created. (Vulkan {})", fmt::ptr(VkDevice(device)),
                    format_vk_api_version(physical_device->get_vk_api_version()));
    }

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
                                          enabled_extensions, enabled_features,
                                          physical_device->get_properties())) {
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
