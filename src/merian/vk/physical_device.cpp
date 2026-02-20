#include "merian/vk/physical_device.hpp"

#include "fmt/ranges.h"
#include "merian/shader/shader_defines.hpp"
#include "merian/utils/string.hpp"
#include "merian/vk/utils/vulkan_extensions.hpp"
#include "merian/vk/utils/vulkan_spirv.hpp"
#include "spdlog/spdlog.h"

namespace merian {

static bool spirv_extension_supported_by_physical_device(
    const char* extension,
    uint32_t vk_api_version,
    const std::unordered_set<std::string>& supported_extensions) {

    const auto device_extension_deps = get_spirv_extension_requirements(extension, vk_api_version);
    bool all_supported = true;
    for (const auto& dep : device_extension_deps) {
        all_supported &= supported_extensions.contains(dep);
    }
    return all_supported;
}

void PhysicalDevice::determine_device_extension_support() {
    const uint32_t effective_vk_instance_api_version = instance->get_vk_api_version();
    const uint32_t effective_vk_device_api_version = get_vk_api_version();

    std::unordered_set<std::string> all_device_extensions;
    for (const auto& ext : physical_device.enumerateDeviceExtensionProperties()) {
        all_device_extensions.emplace(ext.extensionName.data());
    }

    const auto check_extension_recurse = [&](const auto& self,
                                             const char* ext) -> const DeviceExtensionSupport& {
        const auto it = supported_extensions.find(ext);
        if (it != supported_extensions.end()) {
            return it->second;
        }

        // can happen as part of a dependency
        if (!all_device_extensions.contains(ext)) {
            const auto [ins_it, inserted] = supported_extensions.emplace(
                ext,
                DeviceExtensionSupport{false, "the extension is not advertised as supported", {}});
            return ins_it->second;
        }

        const ExtensionInfo* ext_info = get_extension_info(ext);
        if (ext_info == nullptr) {
            const auto [ins_it, inserted] = supported_extensions.emplace(
                ext, DeviceExtensionSupport{true, "the extension is unknown", {}});
            return ins_it->second;
        }
        assert(ext_info->is_device_extension());

        if (ext_info->dependencies.empty()) {
            const auto [ins_it, inserted] =
                supported_extensions.emplace(ext, DeviceExtensionSupport{true, "", {}});
            return ins_it->second;
        }

        // Check dependency branches in order of increasing dependent extension count
        std::vector<ExtensionDependency> dependencies(ext_info->dependencies.begin(),
                                                      ext_info->dependencies.end());
        std::sort(dependencies.begin(), dependencies.end(),
                  [](const ExtensionDependency& a, const ExtensionDependency& b) -> bool {
                      return a.required_extensions.size() < b.required_extensions.size();
                  });

        std::unordered_set<std::string> unsupported_reasons;
        for (const ExtensionDependency& dep : dependencies) {
            if (dep.required_version > effective_vk_device_api_version) {
                unsupported_reasons.insert(
                    fmt::format("Vulkan API version {} is required",
                                format_vk_api_version(dep.required_version)));
                continue;
            }

            DeviceExtensionSupport support{true};
            for (const ExtensionInfo* dep_ext : dep.required_extensions) {
                if (dep_ext->is_instance_extension()) {
                    if (dep_ext->promoted_to_version > effective_vk_instance_api_version &&
                        !instance->extension_enabled(dep_ext->name)) {
                        support.supported = false;
                        support.missing_instance_extensions.emplace(dep_ext->name);
                    }
                    continue;
                }

                const DeviceExtensionSupport& dep_support = self(self, dep_ext->name);
                if (dep_support.supported) {
                    support.dependencies.insert(dep_ext->name);
                    support.dependencies.insert(dep_support.dependencies.begin(),
                                                dep_support.dependencies.end());
                } else if (!dep_support.missing_instance_extensions.empty()) {
                    support.missing_instance_extensions.insert(
                        dep_support.missing_instance_extensions.begin(),
                        dep_support.missing_instance_extensions.end());
                } else {
                    support.supported = false;
                    support.missing_instance_extensions.clear();
                    unsupported_reasons.insert(
                        fmt::format("requires {}, which is unsupported because {}.", dep_ext->name,
                                    dep_support.info));
                    break;
                }
            }
            if (support.supported) {
                const auto [ins_it, inserted] =
                    supported_extensions.emplace(ext, std::move(support));
                return ins_it->second;
            }
            if (!support.missing_instance_extensions.empty()) {
                support.info = fmt::format("instance extensions {} are required",
                                           fmt::join(support.missing_instance_extensions, ", "));
                const auto [ins_it, inserted] =
                    supported_extensions.emplace(ext, std::move(support));
                return ins_it->second;
            }
        }

        const auto [ins_it, inserted] = supported_extensions.emplace(
            ext, DeviceExtensionSupport{
                     false, fmt::format("{}", fmt::join(unsupported_reasons, " or ")), {}});
        return ins_it->second;
    };

    SPDLOG_DEBUG("checking device extension support...");
    [[maybe_unused]] const auto format_support =
        [](const DeviceExtensionSupport& support) -> std::string {
        std::string support_str = support.supported ? "supported" : "unsupported";
        if (!support.info.empty()) {
            support_str += fmt::format(" ({})", support.info);
        }
        if (!support.dependencies.empty()) {
            support_str +=
                fmt::format(" dependencies: [{}]", fmt::join(support.dependencies, ", "));
        }
        if (!support.missing_instance_extensions.empty()) {
            support_str += fmt::format(" missing instance extensions: [{}]",
                                       fmt::join(support.missing_instance_extensions, ", "));
        }
        return support_str;
    };
    for (const auto& ext_props : physical_device.enumerateDeviceExtensionProperties()) {
        const char* const ext = ext_props.extensionName.data();
        const DeviceExtensionSupport& support =
            check_extension_recurse(check_extension_recurse, ext);
        if (support.supported) {
            supported_extension_names.emplace(ext);
        }
        SPDLOG_DEBUG("{} {}", ext, format_support(support));
    }

    // SPDLOG_DEBUG("...with extensions:");
    // const uint32_t device_vk_api_version = physical_device->get_vk_api_version();
    // const uint32_t instance_vk_api_version =
    // physical_device->get_instance()->get_vk_api_version();

    // const auto feature_extensions = enabled_features.get_required_extensions();
    // std::vector<const char*> all_extensions;
    // all_extensions.reserve(additional_extensions.size() + feature_extensions.size());

    // const auto add_extension_recurse =
    //     [&](const auto& self, const ExtensionInfo* ext_info) -> std::pair<bool, std::string> {
    //     if (ext_info->is_instance_extension()) {
    //         // Check if promoted to instance's API version
    //         if (ext_info->promoted_to_version <= instance_vk_api_version) {
    //             return {true, ""}; // Instance extension is promoted, no need to enable
    //         }
    //         if (physical_device->get_instance()->extension_enabled(ext_info->name)) {
    //             return {true, ""};
    //         }
    //         return {false, fmt::format("instance extension {} is not enabled!", ext_info->name)};
    //     }

    //     // already enabled or not necessary
    //     if (enabled_extensions.contains(ext_info->name)) {
    //         return {true, ""};
    //     }
    //     if (ext_info->promoted_to_version <= device_vk_api_version) {
    //         SPDLOG_DEBUG("{} skipped (provided by API version)", ext_info->name);
    //         return {true, ""};
    //     }
    //     if (ext_info->deprecated_by != nullptr &&
    //         physical_device->extension_supported(ext_info->deprecated_by->name)) {
    //         SPDLOG_DEBUG("{} skipped (deprecated by {})", ext_info->name,
    //                      ext_info->deprecated_by->name);
    //         return self(self, ext_info->deprecated_by);
    //     }

    //     if (!physical_device->extension_supported(ext_info->name)) {
    //         return {false, fmt::format("{} not supported by physical device!", ext_info->name)};
    //     }

    //     // Check dependencies - at least one dependency branch must be satisfiable
    //     std::vector<ExtensionDependency> dependencies(ext_info->dependencies.begin(),
    //                                                   ext_info->dependencies.end());
    //     std::sort(dependencies.begin(), dependencies.end(),
    //               [](const ExtensionDependency& a, const ExtensionDependency& b) -> bool {
    //                   return a.required_extensions.size() < b.required_extensions.size();
    //               });
    //     for (const ExtensionDependency& dep : dependencies) {

    //         // For each required extension in this dependency branch
    //         for (const ExtensionInfo* req_ext : dep.required_extensions) {
    //             auto [deps_supported, reason] = self(self, req_ext);
    //             if (!deps_supported) {
    //                 // This branch can't be satisfied, try next one
    //                 // (For now, we just warn but don't fail the whole extension)
    //                 SPDLOG_DEBUG("dependency branch requires {} which is not supported: {}",
    //                              req_ext->name, reason);
    //             }
    //         }
    //     }

    //     enabled_extensions.emplace(ext_info->name);
    //     all_extensions.emplace_back(ext_info->name);
    //     SPDLOG_DEBUG(ext_info->name);
    //     return {true, ""};
    // };

    // const auto add_extension = [&](const auto* const ext) {
    //     if (enabled_extensions.contains(ext)) {
    //         return;
    //     }
    //     auto [supported, reason] =
    //         add_extension_recurse(add_extension_recurse, get_extension_info(ext));
    //     if (!supported) {
    //         SPDLOG_WARN("{} requested but not supported, reason: {}", ext, reason);
    //     }
    // };

    // for (const auto* const ext : additional_extensions) {
    //     add_extension(ext);
    // }
    // for (const auto* const ext : feature_extensions) {
    //     add_extension(ext);
    // }
}

PhysicalDevice::PhysicalDevice(const InstanceHandle& instance,
                               const vk::PhysicalDevice& physical_device)
    : instance(instance), physical_device(physical_device), properties(physical_device, instance),
      supported_features(physical_device, properties) {

    // Extension support
    determine_device_extension_support();

    // Query additional properties
    physical_device_memory_properties = physical_device.getMemoryProperties2();
    physical_device_extension_properties = physical_device.enumerateDeviceExtensionProperties();
    if (supported_extensions.contains(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        cooperative_matrix_properties = physical_device.getCooperativeMatrixPropertiesKHR();
    }

    // Determine SPIRV support
    for (const auto& ext : get_spirv_extensions()) {
        if (spirv_extension_supported_by_physical_device(ext, get_vk_api_version(),
                                                         supported_extension_names)) {
            supported_spirv_extensions.insert(ext);
        }
    }
    for (const auto& cap : get_spirv_capabilities()) {
        if (is_spirv_capability_supported(cap, get_vk_api_version(), supported_extension_names,
                                          supported_features, properties)) {
            supported_spirv_capabilities.insert(cap);
        }
    }

    // Precompute shader defines
    for (const auto& ext : instance->get_enabled_extensions()) {
        shader_defines.emplace(std::string(SHADER_DEFINE_PREFIX_INSTANCE_EXT) + ext, "1");
    }
    for (const auto& ext : supported_extension_names) {
        shader_defines.emplace(std::string(SHADER_DEFINE_PREFIX_DEVICE_EXT) + ext, "1");
    }
    for (const auto& ext : supported_spirv_extensions) {
        shader_defines.emplace(std::string(SHADER_DEFINE_PREFIX_SPIRV_EXT) + ext, "1");
    }
    for (const auto& cap : supported_spirv_capabilities) {
        shader_defines.emplace(std::string(SHADER_DEFINE_PREFIX_SPIRV_CAP) + cap, "1");
    }
}

PhysicalDeviceHandle PhysicalDevice::create(const InstanceHandle& instance,
                                            const vk::PhysicalDevice& physical_device) {
    return PhysicalDeviceHandle(new PhysicalDevice(instance, physical_device));
}

const std::unordered_set<std::string>& PhysicalDevice::get_supported_spirv_extensions() const {
    return supported_spirv_extensions;
}

const std::unordered_set<std::string>& PhysicalDevice::get_supported_spirv_capabilities() const {
    return supported_spirv_capabilities;
}

const std::map<std::string, std::string>& PhysicalDevice::get_shader_defines() const {
    return shader_defines;
}

} // namespace merian
