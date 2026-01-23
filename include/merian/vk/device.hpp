#pragma once

#include "merian/vk/physical_device.hpp"

#include <memory>

namespace merian {

class Device : public std::enable_shared_from_this<Device> {
  private:
    Device(const PhysicalDeviceHandle& physical_device, const vk::Device& device);

  public:
    static DeviceHandle create(const PhysicalDeviceHandle& physical_device,
                               const vk::Device& device);

    ~Device();

    const vk::PipelineCache& get_pipeline_cache() const {
        return pipeline_cache;
    }

    const vk::Device& get_device() const {
        return device;
    }

    const vk::Device& operator*() const {
        return device;
    }

    operator const vk::Device&() const {
        return device;
    }

  private:
    const PhysicalDeviceHandle physical_device;
    const vk::Device device;

    vk::PipelineCache pipeline_cache;
};

using DeviceHandle = std::shared_ptr<Device>;

} // namespace merian
