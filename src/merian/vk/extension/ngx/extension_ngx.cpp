#include "merian/vk/extension/ngx/extension_ngx.hpp"

#ifdef MERIAN_NGX_ENABLED
#include "merian/vk/physical_device.hpp"
#include "ngx.hpp"
#endif

#include <spdlog/spdlog.h>

namespace merian {

#ifdef MERIAN_NGX_ENABLED

ExtensionNGX::~ExtensionNGX() {
    if (!device) {
        return;
    }
    if (capability_params != nullptr) {
        NVSDK_NGX_VULKAN_DestroyParameters(capability_params);
    }
    if (const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Shutdown1(device->get_device());
        NVSDK_NGX_FAILED(result)) {
        SPDLOG_WARN("NGX shutdown failed: {}", ngx_result_string(result));
    }
}

InstanceSupportInfo
ExtensionNGX::query_instance_support([[maybe_unused]] const InstanceSupportQueryInfo& query_info) {
    return {};
}

DeviceSupportInfo ExtensionNGX::query_device_support(const DeviceSupportQueryInfo& query_info) {
    if (!query_info.physical_device->is_nvidia()) {
        return {false, "not an NVIDIA device"};
    }
    return {};
}

void ExtensionNGX::on_device_created(
    const DeviceHandle& device, [[maybe_unused]] const ExtensionContainer& extension_container) {
    const PhysicalDeviceHandle& physical_device = device->get_physical_device();
    const NVSDK_NGX_Application_Identifier identifier = ngx_application_identifier();

    if (const NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
            identifier.v.ProjectDesc.ProjectId, identifier.v.ProjectDesc.EngineType,
            identifier.v.ProjectDesc.EngineVersion, ngx_application_data_path().c_str(),
            physical_device->get_instance()->get_instance(), physical_device->get_physical_device(),
            device->get_device(), VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
            VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr, &ngx_feature_common_info());
        NVSDK_NGX_FAILED(result)) {
        unsupported_reason = fmt::format("NGX init failed: {}", ngx_result_string(result));
        SPDLOG_WARN("{}", unsupported_reason);
        return;
    }
    this->device = device;

    if (const NVSDK_NGX_Result result =
            NVSDK_NGX_VULKAN_GetCapabilityParameters(&capability_params);
        NVSDK_NGX_FAILED(result)) {
        unsupported_reason =
            fmt::format("NGX capability query failed: {}", ngx_result_string(result));
        SPDLOG_WARN("{}", unsupported_reason);
        return;
    }
    unsupported_reason.clear();
}

#else

namespace {
constexpr const char* NOT_BUILT = "merian was built without the NGX SDK (dlss option)";
} // namespace

ExtensionNGX::~ExtensionNGX() = default;

InstanceSupportInfo
ExtensionNGX::query_instance_support([[maybe_unused]] const InstanceSupportQueryInfo& query_info) {
    return {false, NOT_BUILT};
}

DeviceSupportInfo
ExtensionNGX::query_device_support([[maybe_unused]] const DeviceSupportQueryInfo& query_info) {
    return {false, NOT_BUILT};
}

void ExtensionNGX::on_device_created(
    [[maybe_unused]] const DeviceHandle& device,
    [[maybe_unused]] const ExtensionContainer& extension_container) {}

#endif

void ExtensionNGX::on_unsupported(const std::string& reason) {
    SPDLOG_DEBUG("extension {} not supported ({})", name, reason);
}

} // namespace merian
