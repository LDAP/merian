#include "merian/vk/instance.hpp"
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
                             instance_create_info.enabledExtensionCount) {}

InstanceHandle Instance::create(const vk::InstanceCreateInfo& instance_create_info) {
    return InstanceHandle(new Instance(instance_create_info));
}

Instance::~Instance() {
    SPDLOG_DEBUG("destroy instance");
    instance.destroy();
}

} // namespace merian
