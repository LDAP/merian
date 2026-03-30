#include "merian/vk/imgui/imgui_input_listener.hpp"
#include "merian/vk/imgui/imgui_context.hpp"

#include <spdlog/spdlog.h>

namespace merian {

namespace {

// clang-format off
ImGuiKey imgui_key_from_merian(const InputController::Key key) {
    using K = InputController::Key;
    switch (key) {
    case K::TAB:           return ImGuiKey_Tab;
    case K::LEFT:          return ImGuiKey_LeftArrow;
    case K::RIGHT:         return ImGuiKey_RightArrow;
    case K::UP:            return ImGuiKey_UpArrow;
    case K::DOWN:          return ImGuiKey_DownArrow;
    case K::PAGE_UP:       return ImGuiKey_PageUp;
    case K::PAGE_DOWN:     return ImGuiKey_PageDown;
    case K::HOME:          return ImGuiKey_Home;
    case K::END:           return ImGuiKey_End;
    case K::INSERT:        return ImGuiKey_Insert;
    case K::DELETE_KEY:        return ImGuiKey_Delete;
    case K::BACKSPACE:     return ImGuiKey_Backspace;
    case K::SPACE:         return ImGuiKey_Space;
    case K::ENTER:         return ImGuiKey_Enter;
    case K::ESCAPE:        return ImGuiKey_Escape;
    case K::APOSTROPHE:    return ImGuiKey_Apostrophe;
    case K::COMMA:         return ImGuiKey_Comma;
    case K::MINUS:         return ImGuiKey_Minus;
    case K::PERIOD:        return ImGuiKey_Period;
    case K::SLASH:         return ImGuiKey_Slash;
    case K::SEMICOLON:     return ImGuiKey_Semicolon;
    case K::EQUAL:         return ImGuiKey_Equal;
    case K::LEFT_BRACKET:  return ImGuiKey_LeftBracket;
    case K::BACKSLASH:     return ImGuiKey_Backslash;
    case K::RIGHT_BRACKET: return ImGuiKey_RightBracket;
    case K::GRAVE_ACCENT:  return ImGuiKey_GraveAccent;
    case K::CAPS_LOCK:     return ImGuiKey_CapsLock;
    case K::SCROLL_LOCK:   return ImGuiKey_ScrollLock;
    case K::NUM_LOCK:      return ImGuiKey_NumLock;
    case K::PRINT_SCREEN:  return ImGuiKey_PrintScreen;
    case K::PAUSE:         return ImGuiKey_Pause;
    case K::F1:            return ImGuiKey_F1;
    case K::F2:            return ImGuiKey_F2;
    case K::F3:            return ImGuiKey_F3;
    case K::F4:            return ImGuiKey_F4;
    case K::F5:            return ImGuiKey_F5;
    case K::F6:            return ImGuiKey_F6;
    case K::F7:            return ImGuiKey_F7;
    case K::F8:            return ImGuiKey_F8;
    case K::F9:            return ImGuiKey_F9;
    case K::F10:           return ImGuiKey_F10;
    case K::F11:           return ImGuiKey_F11;
    case K::F12:           return ImGuiKey_F12;
    case K::KP_0:          return ImGuiKey_Keypad0;
    case K::KP_1:          return ImGuiKey_Keypad1;
    case K::KP_2:          return ImGuiKey_Keypad2;
    case K::KP_3:          return ImGuiKey_Keypad3;
    case K::KP_4:          return ImGuiKey_Keypad4;
    case K::KP_5:          return ImGuiKey_Keypad5;
    case K::KP_6:          return ImGuiKey_Keypad6;
    case K::KP_7:          return ImGuiKey_Keypad7;
    case K::KP_8:          return ImGuiKey_Keypad8;
    case K::KP_9:          return ImGuiKey_Keypad9;
    case K::KP_DECIMAL:    return ImGuiKey_KeypadDecimal;
    case K::KP_DIVIDE:     return ImGuiKey_KeypadDivide;
    case K::KP_MULTIPLY:   return ImGuiKey_KeypadMultiply;
    case K::KP_SUBTRACT:   return ImGuiKey_KeypadSubtract;
    case K::KP_ADD:        return ImGuiKey_KeypadAdd;
    case K::KP_ENTER:      return ImGuiKey_KeypadEnter;
    case K::KP_EQUAL:      return ImGuiKey_KeypadEqual;
    case K::LEFT_SHIFT:    return ImGuiKey_LeftShift;
    case K::LEFT_CTRL:     return ImGuiKey_LeftCtrl;
    case K::LEFT_ALT:      return ImGuiKey_LeftAlt;
    case K::LEFT_SUPER:    return ImGuiKey_LeftSuper;
    case K::RIGHT_SHIFT:   return ImGuiKey_RightShift;
    case K::RIGHT_CTRL:    return ImGuiKey_RightCtrl;
    case K::RIGHT_ALT:     return ImGuiKey_RightAlt;
    case K::RIGHT_SUPER:   return ImGuiKey_RightSuper;
    case K::MENU:          return ImGuiKey_Menu;
    case K::NUM_0:         return ImGuiKey_0;
    case K::NUM_1:         return ImGuiKey_1;
    case K::NUM_2:         return ImGuiKey_2;
    case K::NUM_3:         return ImGuiKey_3;
    case K::NUM_4:         return ImGuiKey_4;
    case K::NUM_5:         return ImGuiKey_5;
    case K::NUM_6:         return ImGuiKey_6;
    case K::NUM_7:         return ImGuiKey_7;
    case K::NUM_8:         return ImGuiKey_8;
    case K::NUM_9:         return ImGuiKey_9;
    case K::A:             return ImGuiKey_A;
    case K::B:             return ImGuiKey_B;
    case K::C:             return ImGuiKey_C;
    case K::D:             return ImGuiKey_D;
    case K::E:             return ImGuiKey_E;
    case K::F:             return ImGuiKey_F;
    case K::G:             return ImGuiKey_G;
    case K::H:             return ImGuiKey_H;
    case K::I:             return ImGuiKey_I;
    case K::J:             return ImGuiKey_J;
    case K::K:             return ImGuiKey_K;
    case K::L:             return ImGuiKey_L;
    case K::M:             return ImGuiKey_M;
    case K::N:             return ImGuiKey_N;
    case K::O:             return ImGuiKey_O;
    case K::P:             return ImGuiKey_P;
    case K::Q:             return ImGuiKey_Q;
    case K::R:             return ImGuiKey_R;
    case K::S:             return ImGuiKey_S;
    case K::T:             return ImGuiKey_T;
    case K::U:             return ImGuiKey_U;
    case K::V:             return ImGuiKey_V;
    case K::W:             return ImGuiKey_W;
    case K::X:             return ImGuiKey_X;
    case K::Y:             return ImGuiKey_Y;
    case K::Z:             return ImGuiKey_Z;
    default:               return ImGuiKey_None;
    }
}
// clang-format on

void update_mods(ImGuiIO& io, int mods) {
    io.AddKeyEvent(ImGuiMod_Ctrl, (mods & InputController::CONTROL) != 0);
    io.AddKeyEvent(ImGuiMod_Shift, (mods & InputController::SHIFT) != 0);
    io.AddKeyEvent(ImGuiMod_Alt, (mods & InputController::ALT) != 0);
    io.AddKeyEvent(ImGuiMod_Super, (mods & InputController::SUPER) != 0);
}

} // namespace

ImGuiInputListener::ImGuiInputListener(ImGuiContextHandle ctx) : ctx(std::move(ctx)) {}

bool ImGuiInputListener::on_cursor(InputController& c, const double xpos, const double ypos) {
    if (c.is_mouse_grabbed()) {
        return false;
    }

    ImGuiIO& io = ctx->get_io();
    io.AddMousePosEvent(static_cast<float>(xpos), static_cast<float>(ypos));
    return io.WantCaptureMouse;
}

bool ImGuiInputListener::on_mouse_button(InputController& /*c*/,
                                         const InputController::MouseButton button,
                                         const InputController::KeyStatus status) {
    ImGuiIO& io = ctx->get_io();
    io.AddMouseButtonEvent(static_cast<int>(button), status != InputController::KeyStatus::RELEASE);

    return io.WantCaptureMouse;
}

bool ImGuiInputListener::on_scroll(InputController& /*c*/,
                                   const double xoffset,
                                   const double yoffset) {
    ImGuiIO& io = ctx->get_io();
    io.AddMouseWheelEvent(static_cast<float>(xoffset), static_cast<float>(yoffset));

    return io.WantCaptureMouse;
}

bool ImGuiInputListener::on_key(InputController& /*c*/,
                                const InputController::Key key,
                                const InputController::KeyStatus action,
                                const int mods) {
    ImGuiIO& io = ctx->get_io();
    update_mods(io, mods);
    const ImGuiKey imgui_key = imgui_key_from_merian(key);
    if (imgui_key != ImGuiKey_None)
        io.AddKeyEvent(imgui_key, action != InputController::KeyStatus::RELEASE);

    return io.WantCaptureKeyboard;
}

bool ImGuiInputListener::on_char(InputController& /*c*/, const unsigned int codepoint) {
    ImGuiIO& io = ctx->get_io();
    io.AddInputCharacter(codepoint);

    return io.WantCaptureKeyboard;
}

} // namespace merian
