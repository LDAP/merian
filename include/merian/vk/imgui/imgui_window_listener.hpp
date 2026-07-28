#pragma once

#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/window/window_listener.hpp"

namespace merian {

// Forwards window focus to Dear ImGui. Sizes and scale are polled per frame by the backend.
class ImGuiWindowListener : public WindowListener {
  public:
    explicit ImGuiWindowListener(ImGuiContextHandle ctx);

    void on_focus_changed(bool focused) override;

  private:
    ImGuiContextHandle ctx;
};

} // namespace merian
