#pragma once

#include <memory>
#include <vulkan/vulkan.hpp>

namespace merian {

class WindowListener {
  public:
    virtual ~WindowListener() = default;

    virtual void on_resize(vk::Extent2D /*framebuffer_extent*/, vk::Extent2D /*window_extent*/) {}
    virtual void on_display_scale_changed(float /*display_scale*/) {}
    virtual void on_focus_changed(bool /*focused*/) {}
    virtual void on_minimized() {}
    virtual void on_restored() {}
};

using WindowListenerHandle = std::shared_ptr<WindowListener>;

} // namespace merian
