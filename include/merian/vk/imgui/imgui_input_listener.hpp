#pragma once

#include "merian/utils/input_listener.hpp"
#include "merian/vk/imgui/imgui_context.hpp"

namespace merian {

// Forwards input events from an InputController to a Dear ImGui context.
// Cursor position events are suppressed when raw mouse input is active (coordinates
// are delta-accumulated, not screen-space). All other events pass through unconditionally;
// use listener priority to control whether ImGui or the game sees events first.
class ImGuiInputListener : public InputListener {
  public:
    explicit ImGuiInputListener(ImGuiContextHandle ctx);

    bool on_cursor(InputController& c, double xpos, double ypos) override;
    bool on_mouse_button(InputController& c,
                         InputController::MouseButton button,
                         InputController::KeyStatus status,
                         int mods) override;
    bool on_scroll(InputController& c, double xoffset, double yoffset) override;
    bool on_key(InputController& c,
                InputController::Key key,
                InputController::KeyStatus action,
                int mods) override;
    bool on_char(InputController& c, unsigned int codepoint) override;

  private:
    ImGuiContextHandle ctx;
};

} // namespace merian
