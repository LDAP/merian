#pragma once

#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/window/window_listener.hpp"

namespace merian {

// Updates Dear ImGui DisplaySize/DisplayFramebufferScale on resize or display scale change.
class ImGuiWindowListener : public WindowListener {
  public:
    explicit ImGuiWindowListener(ImGuiContextHandle ctx);

    void on_resize(vk::Extent2D framebuffer_extent, vk::Extent2D window_extent) override;
    void on_display_scale_changed(float display_scale) override;
    void on_focus_changed(bool focused) override;

  private:
    ImGuiContextHandle ctx;
};

} // namespace merian
