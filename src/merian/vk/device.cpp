#include "merian/vk/device.hpp"
#include "spdlog/spdlog.h"

namespace merian {

Device::~Device() {
    device.waitIdle();

    SPDLOG_DEBUG("destroy pipeline cache");
    device.destroyPipelineCache(pipeline_cache);

    SPDLOG_DEBUG("destroy device");
    device.destroy();
}

} // namespace merian
