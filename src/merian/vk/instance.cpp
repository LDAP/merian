#include "merian/vk/instance.hpp"
#include "merian/vk/physical_device.hpp"

#include "spdlog/spdlog.h"

namespace merian {

Instance::Instance(const vk::InstanceCreateInfo& instance_create_info)
    : instance(vk::createInstance(instance_create_info)),
      vk_api_version(instance_create_info.pApplicationInfo->apiVersion),
      enabled_layers(instance_create_info.ppEnabledLayerNames,
                     instance_create_info.ppEnabledLayerNames +
                         instance_create_info.enabledLayerCount),
      enabled_extensions(instance_create_info.ppEnabledExtensionNames,
                         instance_create_info.ppEnabledExtensionNames +
                             instance_create_info.enabledExtensionCount) {
    [[maybe_unused]] const uint32_t instance_vulkan_version = vk::enumerateInstanceVersion();
    SPDLOG_DEBUG("instance ({}) created (version: {}.{}.{})", fmt::ptr(VkInstance(instance)),
                 VK_API_VERSION_MAJOR(instance_vulkan_version),
                 VK_API_VERSION_MINOR(instance_vulkan_version),
                 VK_API_VERSION_PATCH(instance_vulkan_version));
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
