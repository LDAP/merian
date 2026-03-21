#pragma once

#include <memory>
#include <utility>
#include <vector>

namespace merian {

class InputListener;

class InputController {

  public:
    // clang-format off
    enum class MouseButton : uint8_t { MOUSE1, MOUSE2, MOUSE3, MOUSE4, MOUSE5, UNKNOWN };
    enum class KeyStatus   : uint8_t { PRESS, RELEASE, REPEAT, UNKNOWN };
    enum ModKey            : uint8_t {
        SHIFT     = 0x01,
        CONTROL   = 0x02,
        ALT       = 0x04,
        SUPER     = 0x08,
        CAPS_LOCK = 0x10,
        NUM_LOCK  = 0x20
    };

    // Platform-independent key identifiers. Contiguous ranges:
    //   A..Z and NUM_0..NUM_9 must remain contiguous for arithmetic mapping.
    enum class Key : uint8_t {
        UNKNOWN = 0,

        // Printable — symbols
        SPACE, APOSTROPHE, COMMA, MINUS, PERIOD, SLASH,
        SEMICOLON, EQUAL, LEFT_BRACKET, BACKSLASH, RIGHT_BRACKET, GRAVE_ACCENT,

        // Printable — digits (contiguous: NUM_0 … NUM_9)
        NUM_0, NUM_1, NUM_2, NUM_3, NUM_4,
        NUM_5, NUM_6, NUM_7, NUM_8, NUM_9,

        // Printable — letters (contiguous: A … Z)
        A, B, C, D, E, F, G, H, I, J, K, L, M,
        N, O, P, Q, R, S, T, U, V, W, X, Y, Z,

        // Control
        ESCAPE, ENTER, TAB, BACKSPACE, INSERT, DELETE,
        RIGHT, LEFT, DOWN, UP,
        PAGE_UP, PAGE_DOWN, HOME, END,
        CAPS_LOCK, SCROLL_LOCK, NUM_LOCK, PRINT_SCREEN, PAUSE,

        // Function keys
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,

        // Numpad
        KP_0, KP_1, KP_2, KP_3, KP_4, KP_5, KP_6, KP_7, KP_8, KP_9,
        KP_DECIMAL, KP_DIVIDE, KP_MULTIPLY, KP_SUBTRACT, KP_ADD, KP_ENTER, KP_EQUAL,

        // Modifiers
        LEFT_SHIFT, LEFT_CTRL, LEFT_ALT, LEFT_SUPER,
        RIGHT_SHIFT, RIGHT_CTRL, RIGHT_ALT, RIGHT_SUPER,

        MENU,
    };
    // clang-format on

    virtual ~InputController() = default;

    // Grab/ungrab mouse: hide cursor, enable raw/relative input, confine to window.
    // Returns true if supported.
    virtual bool set_mouse_grabbed(bool /*grabbed*/) { return false; }
    // Returns true if the mouse is currently grabbed (cursor hidden, confined, raw input active).
    virtual bool is_mouse_grabbed() const { return false; }

    // Register a listener. Higher priority means it is called first.
    // The controller stores a weak_ptr; expired listeners are removed
    // automatically during dispatch.
    void add_listener(std::weak_ptr<InputListener> listener, int priority = 0);

    // Remove all registered listeners.
    void clear_listeners();

    // Enable or disable event dispatch globally. When inactive, no listeners
    // are called.
    void set_active(bool active);

  protected:
    // Dispatch helpers — call from derived window implementations.
    // Return true if any listener consumed the event.
    bool dispatch_cursor(double xpos, double ypos);
    bool dispatch_mouse_button(MouseButton button, KeyStatus status, int mods);
    bool dispatch_scroll(double xoffset, double yoffset);
    bool dispatch_key(Key key, KeyStatus action, int mods);
    bool dispatch_char(unsigned int codepoint);

  private:
    // Sorted descending by priority (index 0 = highest priority).
    std::vector<std::pair<int, std::weak_ptr<InputListener>>> listeners;
    bool input_active = true;
};

using InputControllerHandle = std::shared_ptr<merian::InputController>;

} // namespace merian
