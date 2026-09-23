#include "merian-graph/nodes/path_debug/path_debug_input.hpp"

#include <utility>

namespace merian {

bool PathDebugInput::on_cursor([[maybe_unused]] InputController& controller,
                               const double xpos,
                               const double ypos) {
    cursor_pos = float2(static_cast<float>(xpos), static_cast<float>(ypos));
    cursor_seen = true;
    return false;
}

bool PathDebugInput::on_key([[maybe_unused]] InputController& controller,
                            const InputController::Key key,
                            const InputController::KeyStatus action,
                            [[maybe_unused]] const int mods) {
    if (key == InputController::Key::ESCAPE && action == InputController::KeyStatus::PRESS) {
        clear_requested = true;
        return true;
    }
    const bool is_modifier =
        (modifier == InputController::ModKey::CONTROL &&
         (key == InputController::Key::LEFT_CTRL || key == InputController::Key::RIGHT_CTRL)) ||
        (modifier == InputController::ModKey::SHIFT &&
         (key == InputController::Key::LEFT_SHIFT || key == InputController::Key::RIGHT_SHIFT)) ||
        (modifier == InputController::ModKey::ALT &&
         (key == InputController::Key::LEFT_ALT || key == InputController::Key::RIGHT_ALT));
    if (is_modifier) {
        modifier_down = action == InputController::KeyStatus::PRESS ||
                        action == InputController::KeyStatus::REPEAT;
    }
    return false;
}

bool PathDebugInput::on_mouse_button([[maybe_unused]] InputController& controller,
                                     const InputController::MouseButton button,
                                     const InputController::KeyStatus status) {
    if (button != InputController::MouseButton::MOUSE1) {
        return false;
    }
    if (status == InputController::KeyStatus::PRESS) {
        click_pos = cursor_pos;
        clicked = true;
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
    if (!modifier_down) {
        return false;
    }
    scroll_accum += yoffset;
    return true;
}

std::optional<float2> PathDebugInput::cursor() const {
    if (!cursor_seen) {
        return std::nullopt;
    }
    return cursor_pos;
}

std::optional<float2> PathDebugInput::take_click() {
    if (!clicked) {
        return std::nullopt;
    }
    clicked = false;
    return click_pos;
}

double PathDebugInput::take_scroll() {
    return std::exchange(scroll_accum, 0.);
}

bool PathDebugInput::take_clear() {
    return std::exchange(clear_requested, false);
}

} // namespace merian
