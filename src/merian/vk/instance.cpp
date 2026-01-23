#include "merian/vk/instance.hpp"
#include "spdlog/spdlog.h"

namespace merian {

Instance::Instance(const vk::Instance& instance) : instance(instance) {}

InstanceHandle Instance::create(const vk::Instance& instance) {
    return InstanceHandle(new Instance(instance));
}

Instance::~Instance() {
    SPDLOG_DEBUG("destroy instance");
    instance.destroy();
}

} // namespace merian
