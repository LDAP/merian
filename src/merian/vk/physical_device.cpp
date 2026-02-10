#include "merian/vk/physical_device.hpp"

#include "merian/shader/shader_defines.hpp"
#include "merian/vk/utils/vulkan_spirv.hpp"

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

PhysicalDevice::PhysicalDevice(const InstanceHandle& instance,
                               const vk::PhysicalDevice& physical_device)
    : instance(instance), physical_device(physical_device), properties(physical_device, instance),
      supported_features(physical_device, properties) {

    for (const auto& ext : physical_device.enumerateDeviceExtensionProperties()) {
        supported_extensions.emplace(ext.extensionName.data());
    }

    physical_device_memory_properties = physical_device.getMemoryProperties2();
    physical_device_extension_properties = physical_device.enumerateDeviceExtensionProperties();

    for (const auto& ext : get_spirv_extensions()) {
        if (spirv_extension_supported_by_physical_device(ext, get_vk_api_version(),
                                                         supported_extensions)) {
            supported_spirv_extensions.insert(ext);
        }
    }

    for (const auto& cap : get_spirv_capabilities()) {
        if (is_spirv_capability_supported(cap, get_vk_api_version(), supported_features,
                                          properties)) {
            supported_spirv_capabilities.insert(cap);
        }
    }

    // Precompute shader defines
    for (const auto& ext : supported_extensions) {
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
