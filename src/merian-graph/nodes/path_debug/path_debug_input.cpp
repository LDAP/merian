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
    case InputController::Key::LEFT_SUPER:
    case InputController::Key::RIGHT_SUPER:
        return InputController::SUPER;
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
    if (key == InputController::Key::BACKSPACE && action == InputController::KeyStatus::PRESS) {
        clear_mods = held_mods | mods;
    }
    return false;
}

bool PathDebugInput::on_key([[maybe_unused]] InputController& controller,
                            const InputController::Key key,
                            [[maybe_unused]] const InputController::KeyStatus action,
                            const int mods) {
    return key == InputController::Key::BACKSPACE && modifier_held(mods);
}

bool PathDebugInput::on_mouse_button([[maybe_unused]] InputController& controller,
                                     const InputController::MouseButton button,
                                     const InputController::KeyStatus status) {
    if (button != InputController::MouseButton::MOUSE1) {
        return false;
    }
    const bool select = modifier_held(0);
    if (status == InputController::KeyStatus::PRESS && (over_panels() || select)) {
        events.click = Click{observed->cursor_pos, select};
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
        events.panel_scroll += yoffset;
        return true;
    }
    if (modifier_held(0)) {
        events.radius_scroll += yoffset;
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

void PathDebugInput::set_modifier(const int modifier) {
    this->modifier = modifier;
}

bool PathDebugInput::modifier_held(const int mods) const {
    return ((observed->held_mods | mods) & modifier) != 0;
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

PathDebugInput::Events PathDebugInput::consume_events() {
    const std::optional<int> mods = std::exchange(observed->clear_mods, std::nullopt);
    events.clear = mods && (*mods & modifier) != 0;
    return std::exchange(events, Events{});
}

} // namespace merian
