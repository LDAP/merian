#include "merian/vk/extension/ngx/extension_dlss.hpp"

#include "merian/vk/extension/extension_registry.hpp"

#ifdef MERIAN_NGX_ENABLED
#include "merian/vk/physical_device.hpp"
#include "ngx.hpp"

#include <mutex>
#include <unordered_set>
#endif

#include <spdlog/spdlog.h>

namespace merian {

ExtensionDLSS::ExtensionDLSS(const bool ray_reconstruction)
    : ray_reconstruction(ray_reconstruction) {}

ExtensionDLSSSuperSampling::ExtensionDLSSSuperSampling() : ExtensionDLSS(false) {}

ExtensionDLSSRayReconstruction::ExtensionDLSSRayReconstruction() : ExtensionDLSS(true) {}

std::vector<std::string> ExtensionDLSS::request_extensions() {
    return {ExtensionNGX::name};
}

void ExtensionDLSS::on_unsupported(const std::string& reason) {
    SPDLOG_DEBUG("extension {} not supported ({})",
                 ExtensionRegistry::get_instance().get_name(this), reason);
}

#ifdef MERIAN_NGX_ENABLED

namespace {

NVSDK_NGX_FeatureDiscoveryInfo discovery_info(const bool ray_reconstruction) {
    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID =
        ray_reconstruction ? NVSDK_NGX_Feature_RayReconstruction : NVSDK_NGX_Feature_SuperSampling;
    info.Identifier = ngx_application_identifier();
    info.ApplicationDataPath = ngx_application_data_path().c_str();
    info.FeatureInfo = &ngx_feature_common_info();
    return info;
}

// NGX owns the extension name arrays it hands out.
const char* intern(const char* name) {
    static std::mutex mutex;
    static std::unordered_set<std::string> names;
    const std::lock_guard lock(mutex);
    return names.emplace(name).first->c_str();
}

std::string feature_unsupported_reason(const NVSDK_NGX_FeatureRequirement& requirement) {
    std::vector<std::string> reasons;
    if ((requirement.FeatureSupported & NVSDK_NGX_FeatureSupportResult_DriverVersionUnsupported) !=
        0) {
        reasons.emplace_back("the driver is too old");
    }
    if ((requirement.FeatureSupported & NVSDK_NGX_FeatureSupportResult_AdapterUnsupported) != 0) {
        reasons.emplace_back("the device is not an RTX GPU");
    }
    if ((requirement.FeatureSupported &
         NVSDK_NGX_FeatureSupportResult_OSVersionBelowMinimumSupported) != 0) {
        reasons.emplace_back(fmt::format("the OS is below {}", requirement.MinOSVersion));
    }
    if ((requirement.FeatureSupported & NVSDK_NGX_FeatureSupportResult_NotImplemented) != 0) {
        reasons.emplace_back("the feature is not implemented");
    }
    if (reasons.empty()) {
        reasons.emplace_back(fmt::format("unsupported (0x{:x})",
                                         static_cast<uint32_t>(requirement.FeatureSupported)));
    }
    return fmt::format("{}", fmt::join(reasons, ", "));
}

} // namespace

InstanceSupportInfo
ExtensionDLSS::query_instance_support(const InstanceSupportQueryInfo& query_info) {
    ngx = query_info.extension_container.get_context_extension<ExtensionNGX>();

    const NVSDK_NGX_FeatureDiscoveryInfo discovery = discovery_info(ray_reconstruction);
    uint32_t extension_count = 0;
    VkExtensionProperties* extensions = nullptr;
    if (const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
            &discovery, &extension_count, &extensions);
        NVSDK_NGX_FAILED(result)) {
        return {false, fmt::format("could not query the NGX instance extensions: {}",
                                   ngx_result_string(result))};
    }

    InstanceSupportInfo info;
    for (uint32_t i = 0; i < extension_count; i++) {
        if (!query_info.supported_extensions.contains(extensions[i].extensionName)) {
            return {false,
                    fmt::format("instance extension {} is missing", extensions[i].extensionName)};
        }
        info.required_extensions.emplace_back(intern(extensions[i].extensionName));
    }
    return info;
}

DeviceSupportInfo ExtensionDLSS::query_device_support(const DeviceSupportQueryInfo& query_info) {
    if (DeviceSupportInfo ngx_support = ngx->query_device_support(query_info);
        !ngx_support.supported) {
        return ngx_support;
    }

    const PhysicalDeviceHandle& physical_device = query_info.physical_device;
    const vk::Instance instance = physical_device->get_instance()->get_instance();
    const NVSDK_NGX_FeatureDiscoveryInfo discovery = discovery_info(ray_reconstruction);

    NVSDK_NGX_FeatureRequirement requirement{};
    if (const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureRequirements(
            instance, physical_device->get_physical_device(), &discovery, &requirement);
        NVSDK_NGX_FAILED(result)) {
        return {false, fmt::format("could not query the NGX feature requirements: {}",
                                   ngx_result_string(result))};
    }
    if (requirement.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported) {
        return {false, feature_unsupported_reason(requirement)};
    }

    uint32_t extension_count = 0;
    VkExtensionProperties* extensions = nullptr;
    if (const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            instance, physical_device->get_physical_device(), &discovery, &extension_count,
            &extensions);
        NVSDK_NGX_FAILED(result)) {
        return {false, fmt::format("could not query the NGX device extensions: {}",
                                   ngx_result_string(result))};
    }

    DeviceSupportInfo info;
    for (uint32_t i = 0; i < extension_count; i++) {
        if (!physical_device->extension_supported(extensions[i].extensionName)) {
            return {false,
                    fmt::format("device extension {} is missing", extensions[i].extensionName)};
        }
        info.required_extensions.emplace_back(intern(extensions[i].extensionName));
    }
    return info;
}

void ExtensionDLSS::on_device_created(
    [[maybe_unused]] const DeviceHandle& device,
    [[maybe_unused]] const ExtensionContainer& extension_container) {
    if (!ngx->get_unsupported_reason().empty()) {
        unsupported_reason = ngx->get_unsupported_reason();
    } else {
        // the library reports its own availability only once loaded
        NVSDK_NGX_Parameter* const capability_params = ngx->get_capability_parameters();
        int available = 0;
        NVSDK_NGX_Parameter_GetI(capability_params,
                                 ray_reconstruction
                                     ? NVSDK_NGX_Parameter_SuperSamplingDenoising_Available
                                     : NVSDK_NGX_Parameter_SuperSampling_Available,
                                 &available);
        int init_result = static_cast<int>(NVSDK_NGX_Result_Fail);
        NVSDK_NGX_Parameter_GetI(capability_params,
                                 ray_reconstruction
                                     ? NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult
                                     : NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
                                 &init_result);
        unsupported_reason =
            available != 0 ? "" : ngx_result_string(static_cast<NVSDK_NGX_Result>(init_result));
    }

    SPDLOG_INFO("{}: {}", ExtensionRegistry::get_instance().get_name(this),
                unsupported_reason.empty() ? "available" : unsupported_reason);
}

std::optional<DLSSResolution> ExtensionDLSS::query_resolution(const vk::Extent2D& target_extent,
                                                              const DLSSQuality quality) const {
    NVSDK_NGX_Parameter* const capability_params = ngx->get_capability_parameters();
    if (capability_params == nullptr) {
        return std::nullopt;
    }

    DLSSResolution resolution{};
    float sharpness = 0.f;
    const auto get_optimal_settings =
        ray_reconstruction ? NGX_DLSSD_GET_OPTIMAL_SETTINGS : NGX_DLSS_GET_OPTIMAL_SETTINGS;
    if (const NVSDK_NGX_Result result = get_optimal_settings(
            capability_params, target_extent.width, target_extent.height, ngx_perf_quality(quality),
            &resolution.optimal.width, &resolution.optimal.height, &resolution.max.width,
            &resolution.max.height, &resolution.min.width, &resolution.min.height, &sharpness);
        NVSDK_NGX_FAILED(result)) {
        SPDLOG_WARN("DLSS optimal settings query failed: {}", ngx_result_string(result));
        return std::nullopt;
    }
    if (resolution.optimal.width == 0 || resolution.optimal.height == 0) {
        return std::nullopt;
    }
    return resolution;
}

uint64_t ExtensionDLSS::get_allocated_memory() const {
    NVSDK_NGX_Parameter* const capability_params = ngx->get_capability_parameters();
    if (capability_params == nullptr) {
        return 0;
    }
    unsigned long long bytes = 0;
    const auto get_stats = ray_reconstruction ? NGX_DLSSD_GET_STATS : NGX_DLSS_GET_STATS;
    if (NVSDK_NGX_FAILED(get_stats(capability_params, &bytes))) {
        return 0;
    }
    return bytes;
}

DLSSHandle ExtensionDLSS::create(const DLSSCreateInfo& create_info,
                                 const CommandBufferHandle& cmd) const {
    return DLSSHandle(new DLSS(ngx, ray_reconstruction, create_info, cmd));
}

#else

namespace {
constexpr const char* NOT_BUILT = "merian was built without the NGX SDK (dlss option)";
} // namespace

InstanceSupportInfo
ExtensionDLSS::query_instance_support(const InstanceSupportQueryInfo& query_info) {
    ngx = query_info.extension_container.get_context_extension<ExtensionNGX>();
    return {};
}

DeviceSupportInfo ExtensionDLSS::query_device_support(const DeviceSupportQueryInfo& query_info) {
    return ngx->query_device_support(query_info);
}

void ExtensionDLSS::on_device_created(
    [[maybe_unused]] const DeviceHandle& device,
    [[maybe_unused]] const ExtensionContainer& extension_container) {}

std::optional<DLSSResolution>
ExtensionDLSS::query_resolution([[maybe_unused]] const vk::Extent2D& target_extent,
                                [[maybe_unused]] const DLSSQuality quality) const {
    return std::nullopt;
}

uint64_t ExtensionDLSS::get_allocated_memory() const {
    return 0;
}

DLSSHandle ExtensionDLSS::create([[maybe_unused]] const DLSSCreateInfo& create_info,
                                 [[maybe_unused]] const CommandBufferHandle& cmd) const {
    throw MerianException{NOT_BUILT};
}

#endif

} // namespace merian
