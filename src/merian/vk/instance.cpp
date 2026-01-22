#include "merian/vk/instance.hpp"
#include "spdlog/spdlog.h"

namespace merian {

Instance::Instance(vk::Instance instance) : instance(instance) {}

Instance::~Instance() {
    SPDLOG_DEBUG("destroy instance");
    instance.destroy();
}

} // namespace merian
