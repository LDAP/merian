#pragma once

#include <memory>
#include <vulkan/vulkan.hpp>

namespace merian {

class Window : public std::enable_shared_from_this<Window> {
  public:
    virtual vk::Extent2D framebuffer_extent() = 0;
};

using WindowHandle = std::shared_ptr<Window>;

} // namespace merian
