#pragma once

#include <functional>

namespace merian {

class InputController {

  public:
    enum MouseButton { MOUSE1, MOUSE2, MOUSE3, MOUSE4, MOUSE5 };
    enum KeyStatus { PRESS, RELEASE, REPEAT };
    enum ModKey {
        SHIFT = 0x0001,
        CONTROL = 0x0002,
        ALT = 0x0004,
        SUPER = 0x0008,
        CAPS_LOCK = 0x0010,
        NUM_LOCK = 0x0020
    };

    using MouseCursorEventCallback = std::function<void(
        InputController& controller, double xpos, double ypos)>;

    using MouseButtonEventCallback =
        std::function<void(InputController& controller, MouseButton button, merian::InputController::KeyStatus status, int mods)>;

    using KeyEventCallback = std::function<void(
        InputController& controller, int key, int scancode, KeyStatus action, int mods)>;

    using ScrollEventCallback =
        std::function<void(InputController& controller, double xoffset, double yoffset)>;

    virtual ~InputController() {}

    // Request to enable or disable raw mouse input. This hides disables the cursor
    // and allows unlimited movement. Returns true if success.
    virtual bool request_raw_mouse_input(bool enable) = 0;

    // Returns true if raw mouse input is enabled.
    virtual bool get_raw_mouse_input() = 0;

    // Clear all callbacks
    virtual void reset() = 0;

    virtual void set_mouse_cursor_callback(MouseCursorEventCallback cb) = 0;
    virtual void set_mouse_button_callback(MouseButtonEventCallback cb) = 0;
    virtual void set_scroll_event_callback(ScrollEventCallback cb) = 0;
    virtual void set_key_event_callback(KeyEventCallback cb) = 0;
};

} // namespace merian
