#pragma once

#define VK_ENABLE_BETA_EXTENSIONS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

class Context {
  public:
    Context();

  public:
    vk::Instance instance;
};
