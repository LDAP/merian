#pragma once

#include "merian/shader/shader_compile_context.hpp"
#include "merian/utils/vector.hpp"
#include "merian/vk/context.hpp"
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

namespace merian {

class ExtensionContainer;

/**
 * @brief Result of instance-level support query.
 *
 * Contains whether the extension is supported and what instance-level
 * requirements it needs (extensions and validation layers).
 *
 * The extension must guarantee that all required resources are available when it returns true. If
 * it returns false, the extension may still populate the requirements with the resources that would
 * have been needed, for the purpose of generating error messages.
 */
struct InstanceSupportInfo {
    bool supported = true;                          ///< Whether extension is supported
    std::string unsupported_reason{};               ///< Optional custom reason if unsupported
    std::vector<const char*> required_extensions{}; ///< Required instance extensions
    std::vector<const char*> required_layers{};     ///< Required validation layers
};

/**
 * @brief Context for instance-level support queries.
 *
 * Provides information about what instance extensions and layers are available,
 * and access to other loaded extensions for coordination.
 */
struct InstanceSupportQueryInfo {
    const FileLoaderHandle file_loader;
    const std::unordered_set<std::string>& supported_extensions; ///< Available instance extensions
    const std::unordered_set<std::string>& supported_layers;     ///< Available validation layers
    const ExtensionContainer& extension_container;               ///< Access to loaded extensions
};

/**
 * @brief Result of device-level support query.
 *
 * Contains whether the extension is supported on the device and what device-level
 * requirements it needs (features, extensions, SPIR-V capabilities).
 *
 * The extension must guarantee that all required resources are available when it returns true. If
 * it returns false, the extension may still populate the requirements with the resources that would
 * have been needed, for the purpose of generating error messages.
 */
struct DeviceSupportQueryInfo;

struct DeviceSupportInfo {
    bool supported = true;                          ///< Whether extension is supported on device
    std::string unsupported_reason{};               ///< Optional custom reason if unsupported
    std::vector<const char*> required_features{};   ///< Required Vulkan features (by name)
    std::vector<const char*> required_extensions{}; ///< Required device extensions
    std::vector<const char*> required_spirv_capabilities{}; ///< Required SPIR-V capabilities
    std::vector<const char*> required_spirv_extensions{};   ///< Required SPIR-V extensions

    // Checks required and optional requirements against the physical device.
    // Returns supported=false with a reason if any required item is missing.
    // Optional items are included in the requirements only if supported.
    static DeviceSupportInfo check(const DeviceSupportQueryInfo& query_info,
                                   const std::vector<const char*>& required_features = {},
                                   const std::vector<const char*>& optional_features = {},
                                   const std::vector<const char*>& required_extensions = {},
                                   const std::vector<const char*>& optional_extensions = {},
                                   const std::vector<const char*>& required_spirv_capabilities = {},
                                   const std::vector<const char*>& optional_spirv_capabilities = {},
                                   const std::vector<const char*>& required_spirv_extensions = {},
                                   const std::vector<const char*>& optional_spirv_extensions = {});

    // Combines two DeviceSupportInfo: ANDs supported, concatenates reasons and all requirement
    // vectors.
    friend DeviceSupportInfo operator&(const DeviceSupportInfo& a, const DeviceSupportInfo& b) {
        DeviceSupportInfo result;
        result.supported = a.supported && b.supported;

        if (!a.unsupported_reason.empty() && !b.unsupported_reason.empty()) {
            result.unsupported_reason = a.unsupported_reason + "; " + b.unsupported_reason;
        } else if (!a.unsupported_reason.empty()) {
            result.unsupported_reason = a.unsupported_reason;
        } else {
            result.unsupported_reason = b.unsupported_reason;
        }

        result.required_features = a.required_features;
        insert_all(result.required_features, b.required_features);
        result.required_extensions = a.required_extensions;
        insert_all(result.required_extensions, b.required_extensions);
        result.required_spirv_capabilities = a.required_spirv_capabilities;
        insert_all(result.required_spirv_capabilities, b.required_spirv_capabilities);
        result.required_spirv_extensions = a.required_spirv_extensions;
        insert_all(result.required_spirv_extensions, b.required_spirv_extensions);
        return result;
    }

    DeviceSupportInfo& operator&=(const DeviceSupportInfo& other) {
        *this = *this & other;
        return *this;
    }
};

/**
 * @brief Context for device-level support queries.
 *
 * Provides information about the physical device, queue topology,
 * and access to other loaded extensions for coordination.
 */
struct DeviceSupportQueryInfo {
    const FileLoaderHandle file_loader;
    const PhysicalDeviceHandle& physical_device;
    const QueueInfo& queue_info;
    const ExtensionContainer& extension_container;
    const ShaderCompileContextHandle compile_context;
};

inline std::string format_as(const InstanceSupportInfo& info) {
    if (info.supported && info.required_extensions.empty() && info.required_layers.empty())
        return "supported";

    std::string result = info.supported ? "supported" : "UNSUPPORTED";
    if (!info.unsupported_reason.empty())
        result += fmt::format(" ({})", info.unsupported_reason);
    if (!info.required_extensions.empty())
        result += fmt::format(", extensions: [{}]", fmt::join(info.required_extensions, ", "));
    if (!info.required_layers.empty())
        result += fmt::format(", layers: [{}]", fmt::join(info.required_layers, ", "));
    return result;
}

inline std::string format_as(const DeviceSupportInfo& info) {
    if (info.supported && info.required_features.empty() && info.required_extensions.empty() &&
        info.required_spirv_capabilities.empty() && info.required_spirv_extensions.empty())
        return "supported";

    std::string result = info.supported ? "supported" : "UNSUPPORTED";
    if (!info.unsupported_reason.empty())
        result += fmt::format(" ({})", info.unsupported_reason);
    if (!info.required_features.empty())
        result += fmt::format(", features: [{}]", fmt::join(info.required_features, ", "));
    if (!info.required_extensions.empty())
        result += fmt::format(", extensions: [{}]", fmt::join(info.required_extensions, ", "));
    if (!info.required_spirv_capabilities.empty())
        result +=
            fmt::format(", spirv caps: [{}]", fmt::join(info.required_spirv_capabilities, ", "));
    if (!info.required_spirv_extensions.empty())
        result +=
            fmt::format(", spirv exts: [{}]", fmt::join(info.required_spirv_extensions, ", "));
    return result;
}

/**
 * @brief      An extension to the Vulkan Context.
 *
 * An extension can enable layers and Vulkan instance and device extensions (_EXT), as well as
 * hook into the context creation process. Extensions are checked for compatibility and the result
 * can be retrived using is_supported().
 *
 * If a extension is determined to be not compatible it is removed from the context and
 * the corresponding on_destroy_* lifecycle methods are called.
 */
class ContextExtension {

    friend Context;

  public:
    ContextExtension() {}

    virtual ~ContextExtension() = 0;

    /**
     * @brief Request other extensions that this extension depends on.
     *
     * Called during context initialization before instance creation. Extensions can request
     * other extensions by name, which will be loaded from the extension registry and have
     * their dependencies resolved recursively.
     *
     * @return Vector of extension names to be loaded (e.g., {"glfw", "vk_debug_utils"})
     */
    virtual std::vector<std::string> request_extensions() {
        return {};
    }

    /**
     * @brief Query instance-level support and requirements.
     *
     * Called during instance creation to determine if this extension is supported and what
     * instance extensions and layers it requires. The extension should check the
     * query_info.supported_extensions and query_info.supported_layers to verify its
     * requirements are available.
     *
     * @param query_info Context containing supported extensions/layers and extension container
     * @return InstanceSupportInfo with supported flag and required extensions/layers
     */
    virtual InstanceSupportInfo
    query_instance_support(const InstanceSupportQueryInfo& /*query_info*/) {
        return InstanceSupportInfo{true};
    }

    /**
     * @brief Query device-level support and requirements.
     *
     * Called during physical device selection to determine if this extension is supported on
     * the device and what device extensions, features, and SPIR-V capabilities it requires.
     * The extension should check the physical_device to verify its requirements are available.
     *
     * @param query_info Context containing physical device, queue info, and extension container
     * @return DeviceSupportInfo with supported flag and required extensions/features/SPIR-V
     */
    virtual DeviceSupportInfo query_device_support(const DeviceSupportQueryInfo& /*query_info*/) {
        return DeviceSupportInfo{true};
    }

    // LIFECYCLE (in order)

    virtual void on_context_initializing([[maybe_unused]] const PFN_vkGetInstanceProcAddr loader,
                                         [[maybe_unused]] const FileLoaderHandle& file_loader,
                                         [[maybe_unused]] const ContextCreateInfo& create_info) {}

    /**
     * Append structs to vkInstanceCreateInfo to enable features of extensions.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* pnext_instance_create_info(void* const p_next) {
        return p_next;
    }

    virtual void on_instance_created(const InstanceHandle& /*unused*/,
                                     const ExtensionContainer& /*extension_container*/) {}

    /* Called after the physical device was select and before extensions are checked for
     * compatibility and check_support is called.*/
    virtual void on_physical_device_selected(const PhysicalDeviceHandle& /*unused*/,
                                             const ExtensionContainer& /*extension_container*/) {}

    virtual bool accept_graphics_queue([[maybe_unused]] const InstanceHandle& instance,
                                       [[maybe_unused]] const PhysicalDeviceHandle& physical_device,
                                       [[maybe_unused]] std::size_t queue_family_index) {
        return true;
    }

    virtual void
    on_extension_support_confirmed([[maybe_unused]] const ExtensionContainer& extension_container) {
    }

    /**
     * Append structs to VkDeviceCreateInfo to enable features of extensions.
     *
     * If a struct should be appended, set pNext of your struct to the supplied pointer,
     * then return a pointer to your struct.
     * If nothing should be appended, return the supplied pointer.
     */
    virtual void* pnext_device_create_info(void* const p_next) {
        return p_next;
    }

    /**
     * Called with the device create info just before createDevice is called.
     *
     * Use to hook into device creation.
     */
    virtual void on_create_device(const PhysicalDeviceHandle& /*physical_device*/,
                                  VulkanFeatures& /*features*/,
                                  std::vector<const char*>& /*extensions*/) {}

    virtual void on_device_created(const DeviceHandle& /*unused*/,
                                   const ExtensionContainer& /*extension_container*/) {}

    /* Called right before context constructor returns. */
    virtual void
    on_context_created([[maybe_unused]] const ContextHandle& context,
                       [[maybe_unused]] const ExtensionContainer& extension_container) {}

    // Called by context if extension was determined as unsupported. The extension might not receive
    // further callbacks.
    virtual void on_unsupported([[maybe_unused]] const std::string& reason);
};

} // namespace merian
