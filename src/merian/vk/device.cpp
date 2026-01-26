#include "merian/vk/device.hpp"
#include "spdlog/spdlog.h"

namespace merian {

Device::Device(const PhysicalDeviceHandle& physical_device, const vk::DeviceCreateInfo& create_info)
    : physical_device(physical_device),
      device(physical_device->get_physical_device().createDevice(create_info)) {

    SPDLOG_DEBUG("create pipeline cache");
    vk::PipelineCacheCreateInfo pipeline_cache_create_info{};
    pipeline_cache = device.createPipelineCache(pipeline_cache_create_info);
}

DeviceHandle Device::create(const PhysicalDeviceHandle& physical_device,
                            const vk::DeviceCreateInfo& create_info) {
    return DeviceHandle(new Device(physical_device, create_info));
}

Device::~Device() {
    device.waitIdle();

    SPDLOG_DEBUG("destroy pipeline cache");
    device.destroyPipelineCache(pipeline_cache);

    SPDLOG_DEBUG("destroy device");
    device.destroy();
}

} // namespace merian
