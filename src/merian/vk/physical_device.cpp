#include "merian/vk/physical_device.hpp"

namespace merian {
PhysicalDevice::PhysicalDevice(const InstanceHandle& instance,
                               const vk::PhysicalDevice& physical_device)
    : instance(instance), physical_device(physical_device) {}

PhysicalDeviceHandle PhysicalDevice::create(const InstanceHandle& instance,
                                            const vk::PhysicalDevice& physical_device) {
    return PhysicalDeviceHandle(new PhysicalDevice(instance, physical_device));
}

} // namespace merian
