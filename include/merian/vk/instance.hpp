#pragma once

#include <memory>

// attempt to set the dynamic dispach launcher as early as possible
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif

#include <vulkan/vulkan.hpp>

namespace merian {

class Instance : public std::enable_shared_from_this<Instance> {
  public:
    Instance(vk::Instance instance);

    ~Instance();

    operator vk::Instance&() {
        return instance;
    }

  private:
    vk::Instance instance;
};

using InstanceHandle = std::shared_ptr<Instance>;

} // namespace merian
