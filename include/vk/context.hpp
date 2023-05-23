#pragma once

#include "vk/extension/extension.hpp"
#include <vulkan/vulkan.hpp>

class Context {
  public:
    Context();
    ~Context();

  private:
    void create_instance();

  public:
    vk::Instance instance;
    std::vector<Extension*> extensions;
};
