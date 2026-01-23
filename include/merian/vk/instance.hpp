#pragma once

#include "merian/fwd.hpp"

#include <memory>

// attempt to set the dynamic dispach launcher as early as possible
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif

#include <vulkan/vulkan.hpp>

namespace merian {

class Instance : public std::enable_shared_from_this<Instance> {
  private:
    Instance(const vk::Instance& instance);

  public:
    static InstanceHandle create(const vk::Instance& instance);

    ~Instance();

    const vk::Instance& get_instance() const {
        return instance;
    }

    const vk::Instance& operator*() const {
        return instance;
    }

    operator const vk::Instance&() const {
        return instance;
    }

  private:
    vk::Instance instance;
};

using InstanceHandle = std::shared_ptr<Instance>;

} // namespace merian
