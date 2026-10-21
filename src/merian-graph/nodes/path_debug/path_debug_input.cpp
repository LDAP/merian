#include "merian-graph/nodes/path_debug/path_debug_input.hpp"

#include <utility>

namespace merian {

namespace {

int modifier_of(const InputController::Key key) {
    switch (key) {
    case InputController::Key::LEFT_SHIFT:
    case InputController::Key::RIGHT_SHIFT:
        return InputController::SHIFT;
    case InputController::Key::LEFT_CTRL:
    case InputController::Key::RIGHT_CTRL:
        return InputController::CONTROL;
    case InputController::Key::LEFT_ALT:
    case InputController::Key::RIGHT_ALT:
        return InputController::ALT;
    default:
        return 0;
    }
}

} // namespace

bool PathDebugInput::on_cursor([[maybe_unused]] InputController& controller,
                               [[maybe_unused]] const double xpos,
                               [[maybe_unused]] const double ypos) {
    observed->cursor_taken = false;
    return false;
}

bool PathDebugInput::Observer::on_cursor([[maybe_unused]] InputController& controller,
                                         const double xpos,
                                         const double ypos) {
    cursor_pos = float2(static_cast<float>(xpos), static_cast<float>(ypos));
    cursor_seen = true;
    cursor_taken = true;
    return false;
}

bool PathDebugInput::Observer::on_key([[maybe_unused]] InputController& controller,
                                      const InputController::Key key,
                                      const InputController::KeyStatus action,
                                      const int mods) {
    // a modifier's own event may report the mods from before it
    if (const int modifier = modifier_of(key); modifier != 0) {
        held_mods = action == InputController::KeyStatus::RELEASE ? held_mods & ~modifier
                                                                  : held_mods | modifier;
    } else {
        held_mods = mods;
    }
    escape_pressed |=
        key == InputController::Key::ESCAPE && action == InputController::KeyStatus::PRESS;
    return false;
}

bool PathDebugInput::on_key([[maybe_unused]] InputController& controller,
                            const InputController::Key key,
                            [[maybe_unused]] const InputController::KeyStatus action,
                            [[maybe_unused]] const int mods) {
    return key == InputController::Key::ESCAPE && consume_escape;
}

bool PathDebugInput::on_mouse_button([[maybe_unused]] InputController& controller,
                                     const InputController::MouseButton button,
                                     const InputController::KeyStatus status) {
    if (button != InputController::MouseButton::MOUSE1) {
        return false;
    }
    const bool pick = (observed->held_mods & PICK_MODIFIER) != 0;
    if (status == InputController::KeyStatus::PRESS && (over_panels() || pick)) {
        click = Click{observed->cursor_pos, pick};
        swallow_release = true;
        return true;
    }
    if (status == InputController::KeyStatus::RELEASE && swallow_release) {
        swallow_release = false;
        return true;
    }
    return false;
}

bool PathDebugInput::on_scroll([[maybe_unused]] InputController& controller,
                               [[maybe_unused]] const double xoffset,
                               const double yoffset) {
    if (over_panels()) {
        panel_scroll += yoffset;
        return true;
    }
    if ((observed->held_mods & WIDEN_MODIFIER) != 0) {
        widen_scroll += yoffset;
        return true;
    }
    return false;
}

InputListenerHandle PathDebugInput::observer() const {
    return observed;
}

void PathDebugInput::set_panel_area(const float2& lower, const float2& upper) {
    panel_lower = lower;
    panel_upper = upper;
}

bool PathDebugInput::over_panels() const {
    const float2& pos = observed->cursor_pos;
    return observed->cursor_seen && pos.x >= panel_lower.x && pos.y >= panel_lower.y &&
           pos.x < panel_upper.x && pos.y < panel_upper.y;
}

std::optional<float2> PathDebugInput::cursor() const {
    if (!observed->cursor_seen || observed->cursor_taken) {
        return std::nullopt;
    }
    return observed->cursor_pos;
}

std::optional<PathDebugInput::Click> PathDebugInput::take_click() {
    return std::exchange(click, std::nullopt);
}

double PathDebugInput::take_panel_scroll() {
    return std::exchange(panel_scroll, 0.);
}

double PathDebugInput::take_widen_scroll() {
    return std::exchange(widen_scroll, 0.);
}

bool PathDebugInput::take_escape() {
    return std::exchange(observed->escape_pressed, false);
}

void PathDebugInput::set_consume_escape(const bool consume) {
    consume_escape = consume;
}

} // namespace merian
