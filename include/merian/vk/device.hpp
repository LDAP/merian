#pragma once

#include "merian/vk/physical_device.hpp"

#include <memory>

namespace merian {

class Device : public std::enable_shared_from_this<Device> {

  public:
    Device(const PhysicalDeviceHandle& physical_device, const vk::Device& device);

    ~Device();

    operator const vk::Device&() const {
        return device;
    }

  private:
    const PhysicalDeviceHandle physical_device;
    const vk::Device device;

    vk::PipelineCache pipeline_cache;
};

} // namespace merian
