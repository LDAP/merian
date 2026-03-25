#pragma once

#include "merian/vk/imgui/imgui_context.hpp"
#include "merian/vk/window/window.hpp"

#include <memory>

namespace merian {

// Minimal platform backend for ImGui (no window attached).
//
// Usage:
//   auto ctx     = std::make_shared<ImGuiContext>();
//   auto backend = std::make_shared<ImGuiMerianBackend>(ctx);
//   // each frame, before ImGui UI:
//   backend->new_frame(dt_seconds);
class ImGuiMerianBackend {
  public:
    explicit ImGuiMerianBackend(const ImGuiContextHandle& ctx);
    virtual ~ImGuiMerianBackend();

    // Call once per frame before building ImGui UI.
    // delta_time: seconds since last frame (caller owns timing).
    virtual void new_frame(float delta_time);

  protected:
    static WindowCursorShape cursor_shape_from_imgui(int imgui_cursor);

    ImGuiContextHandle ctx;
};

using ImGuiMerianBackendHandle = std::shared_ptr<ImGuiMerianBackend>;

} // namespace merian
