#pragma once

#include "merian/utils/input_listener.hpp"
#include "merian/vk/imgui/imgui_merian_backend.hpp"
#include "merian/vk/window/window.hpp"
#include "merian/vk/window/window_listener.hpp"

#include <memory>

namespace merian {

// Window-attached platform backend for Dear ImGui.
//
// Registers input and window listeners, wires up clipboard and OS cursor shape updates.
// Recreate this object whenever the window changes — do not call attach() on the old instance.
//
// Usage:
//   auto ctx     = std::make_shared<ImGuiContext>();
//   // on window created/recreated:
//   imgui_backend = std::make_shared<ImGuiMerianWindowBackend>(ctx, window);
//   // each frame, before ImGui UI:
//   imgui_backend->new_frame(dt_seconds);
class ImGuiMerianWindowBackend : public ImGuiMerianBackend {
  public:
    ImGuiMerianWindowBackend(const ImGuiContextHandle& ctx,
                             const WindowHandle& window,
                             int input_priority = 10);
    ~ImGuiMerianWindowBackend() override;

    void new_frame(float delta_time) override;

  private:
    WindowHandle window;
    std::shared_ptr<InputListener> input_listener;
    std::shared_ptr<WindowListener> window_listener;
};

using ImGuiMerianWindowBackendHandle = std::shared_ptr<ImGuiMerianWindowBackend>;

} // namespace merian
