#include "merian/vk/instance.hpp"
#include "merian/vk/physical_device.hpp"

#include "spdlog/spdlog.h"

namespace merian {

Instance::Instance(const vk::InstanceCreateInfo& instance_create_info)
    : instance(vk::createInstance(instance_create_info)),
      effective_vk_api_version(std::min(get_instance_vk_api_version(),
                                        instance_create_info.pApplicationInfo->apiVersion)),
      target_vk_api_version(instance_create_info.pApplicationInfo->apiVersion),
      enabled_layers(instance_create_info.ppEnabledLayerNames,
                     instance_create_info.ppEnabledLayerNames +
                         instance_create_info.enabledLayerCount),
      enabled_extensions(instance_create_info.ppEnabledExtensionNames,
                         instance_create_info.ppEnabledExtensionNames +
                             instance_create_info.enabledExtensionCount) {
    const uint32_t vk_instance_version = get_instance_vk_api_version();
    SPDLOG_DEBUG(
        "instance ({}) created (instance version: {}.{}.{}, target version: {}.{}.{}, effective: "
        "{}.{}.{})",
        fmt::ptr(VkInstance(instance)), VK_API_VERSION_MAJOR(vk_instance_version),
        VK_API_VERSION_MINOR(vk_instance_version), VK_API_VERSION_PATCH(vk_instance_version),
        VK_API_VERSION_MAJOR(target_vk_api_version), VK_API_VERSION_MINOR(target_vk_api_version),
        VK_API_VERSION_PATCH(target_vk_api_version), VK_API_VERSION_MAJOR(effective_vk_api_version),
        VK_API_VERSION_MINOR(effective_vk_api_version),
        VK_API_VERSION_PATCH(effective_vk_api_version));

    vk_get_instance_proc_addr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
}

InstanceHandle Instance::create(const vk::InstanceCreateInfo& instance_create_info) {
    return InstanceHandle(new Instance(instance_create_info));
}

Instance::~Instance() {
    SPDLOG_DEBUG("destroy instance");
    instance.destroy();
}

std::vector<PhysicalDeviceHandle> Instance::get_physical_devices() {
    std::vector<PhysicalDeviceHandle> physical_devices;
    const auto shared_instance = shared_from_this();
    for (const auto& physical_device : instance.enumeratePhysicalDevices()) {
        physical_devices.emplace_back(PhysicalDevice::create(shared_instance, physical_device));
    }

    return physical_devices;
}

} // namespace merian
