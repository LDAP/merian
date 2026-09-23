#pragma once

#include "merian/utils/input_listener.hpp"
#include "merian/utils/vector_matrix.hpp"

#include <optional>

namespace merian {

class PathDebugInput : public InputListener {
  public:
    static constexpr int PRIORITY = 5;

    explicit PathDebugInput(
        const InputController::ModKey modifier = InputController::ModKey::CONTROL)
        : modifier(modifier) {}

    bool on_cursor(InputController& controller, double xpos, double ypos) override;
    bool on_key(InputController& controller,
                InputController::Key key,
                InputController::KeyStatus action,
                int mods) override;
    bool on_mouse_button(InputController& controller,
                         InputController::MouseButton button,
                         InputController::KeyStatus status) override;
    bool on_scroll(InputController& controller, double xoffset, double yoffset) override;

    std::optional<float2> cursor() const;
    std::optional<float2> take_click();
    double take_scroll();
    bool take_clear();

  private:
    const InputController::ModKey modifier;
    bool cursor_seen = false;
    float2 cursor_pos{};
    float2 click_pos{};
    bool modifier_down = false;
    double scroll_accum = 0.;
    bool clicked = false;
    bool clear_requested = false;
    bool swallow_release = false;
};

} // namespace merian
