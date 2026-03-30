#include "merian/vk/extension/sdl/sdl_window.hpp"
#include "merian/vk/extension/sdl/sdl_surface.hpp"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

namespace merian {

namespace {

static constexpr const char* SDL_WINDOW_USER_DATA_KEY = "merian_sdl_window";

InputController::Key key_from_sdl(const SDL_Keycode keycode, const SDL_Scancode scancode) {
    using K = InputController::Key;

    // Keypad doesn't have individual key values in SDL3
    switch (scancode) {
    case SDL_SCANCODE_KP_0:
        return K::KP_0;
    case SDL_SCANCODE_KP_1:
        return K::KP_1;
    case SDL_SCANCODE_KP_2:
        return K::KP_2;
    case SDL_SCANCODE_KP_3:
        return K::KP_3;
    case SDL_SCANCODE_KP_4:
        return K::KP_4;
    case SDL_SCANCODE_KP_5:
        return K::KP_5;
    case SDL_SCANCODE_KP_6:
        return K::KP_6;
    case SDL_SCANCODE_KP_7:
        return K::KP_7;
    case SDL_SCANCODE_KP_8:
        return K::KP_8;
    case SDL_SCANCODE_KP_9:
        return K::KP_9;
    case SDL_SCANCODE_KP_PERIOD:
        return K::KP_DECIMAL;
    case SDL_SCANCODE_KP_DIVIDE:
        return K::KP_DIVIDE;
    case SDL_SCANCODE_KP_MULTIPLY:
        return K::KP_MULTIPLY;
    case SDL_SCANCODE_KP_MINUS:
        return K::KP_SUBTRACT;
    case SDL_SCANCODE_KP_PLUS:
        return K::KP_ADD;
    case SDL_SCANCODE_KP_ENTER:
        return K::KP_ENTER;
    case SDL_SCANCODE_KP_EQUALS:
        return K::KP_EQUAL;
    default:
        break;
    }

    switch (keycode) {
    case SDLK_TAB:
        return K::TAB;
    case SDLK_LEFT:
        return K::LEFT;
    case SDLK_RIGHT:
        return K::RIGHT;
    case SDLK_UP:
        return K::UP;
    case SDLK_DOWN:
        return K::DOWN;
    case SDLK_PAGEUP:
        return K::PAGE_UP;
    case SDLK_PAGEDOWN:
        return K::PAGE_DOWN;
    case SDLK_HOME:
        return K::HOME;
    case SDLK_END:
        return K::END;
    case SDLK_INSERT:
        return K::INSERT;
    case SDLK_DELETE:
        return K::DELETE_KEY;
    case SDLK_BACKSPACE:
        return K::BACKSPACE;
    case SDLK_SPACE:
        return K::SPACE;
    case SDLK_RETURN:
        return K::ENTER;
    case SDLK_ESCAPE:
        return K::ESCAPE;
    // case SDLK_APOSTROPHE: return K::APOSTROPHE;
    case SDLK_COMMA:
        return K::COMMA;
    // case SDLK_MINUS: return K::MINUS;
    case SDLK_PERIOD:
        return K::PERIOD;
    // case SDLK_SLASH: return K::SLASH;
    case SDLK_SEMICOLON:
        return K::SEMICOLON;
    // case SDLK_EQUALS: return K::EQUAL;
    // case SDLK_LEFTBRACKET: return K::LEFTBRACKET;
    // case SDLK_BACKSLASH: return K::BACKSLASH;
    // case SDLK_RIGHTBRACKET: return K::RIGHTBRACKET;
    // case SDLK_GRAVE: return K::GRAVEACCENT;
    case SDLK_CAPSLOCK:
        return K::CAPS_LOCK;
    case SDLK_SCROLLLOCK:
        return K::SCROLL_LOCK;
    case SDLK_NUMLOCKCLEAR:
        return K::NUM_LOCK;
    case SDLK_PRINTSCREEN:
        return K::PRINT_SCREEN;
    case SDLK_PAUSE:
        return K::PAUSE;
    case SDLK_LCTRL:
        return K::LEFT_CTRL;
    case SDLK_LSHIFT:
        return K::LEFT_SHIFT;
    case SDLK_LALT:
        return K::LEFT_ALT;
    case SDLK_LGUI:
        return K::LEFT_SUPER;
    case SDLK_RCTRL:
        return K::RIGHT_CTRL;
    case SDLK_RSHIFT:
        return K::RIGHT_SHIFT;
    case SDLK_RALT:
        return K::RIGHT_ALT;
    case SDLK_RGUI:
        return K::RIGHT_SUPER;
    case SDLK_APPLICATION:
        return K::MENU;
    case SDLK_0:
        return K::NUM_0;
    case SDLK_1:
        return K::NUM_1;
    case SDLK_2:
        return K::NUM_2;
    case SDLK_3:
        return K::NUM_3;
    case SDLK_4:
        return K::NUM_4;
    case SDLK_5:
        return K::NUM_5;
    case SDLK_6:
        return K::NUM_6;
    case SDLK_7:
        return K::NUM_7;
    case SDLK_8:
        return K::NUM_8;
    case SDLK_9:
        return K::NUM_9;
    case SDLK_A:
        return K::A;
    case SDLK_B:
        return K::B;
    case SDLK_C:
        return K::C;
    case SDLK_D:
        return K::D;
    case SDLK_E:
        return K::E;
    case SDLK_F:
        return K::F;
    case SDLK_G:
        return K::G;
    case SDLK_H:
        return K::H;
    case SDLK_I:
        return K::I;
    case SDLK_J:
        return K::J;
    case SDLK_K:
        return K::K;
    case SDLK_L:
        return K::L;
    case SDLK_M:
        return K::M;
    case SDLK_N:
        return K::N;
    case SDLK_O:
        return K::O;
    case SDLK_P:
        return K::P;
    case SDLK_Q:
        return K::Q;
    case SDLK_R:
        return K::R;
    case SDLK_S:
        return K::S;
    case SDLK_T:
        return K::T;
    case SDLK_U:
        return K::U;
    case SDLK_V:
        return K::V;
    case SDLK_W:
        return K::W;
    case SDLK_X:
        return K::X;
    case SDLK_Y:
        return K::Y;
    case SDLK_Z:
        return K::Z;
    case SDLK_F1:
        return K::F1;
    case SDLK_F2:
        return K::F2;
    case SDLK_F3:
        return K::F3;
    case SDLK_F4:
        return K::F4;
    case SDLK_F5:
        return K::F5;
    case SDLK_F6:
        return K::F6;
    case SDLK_F7:
        return K::F7;
    case SDLK_F8:
        return K::F8;
    case SDLK_F9:
        return K::F9;
    case SDLK_F10:
        return K::F10;
    case SDLK_F11:
        return K::F11;
    case SDLK_F12:
        return K::F12;
    // case SDLK_F13: return K::F13;
    // case SDLK_F14: return K::F14;
    // case SDLK_F15: return K::F15;
    // case SDLK_F16: return K::F16;
    // case SDLK_F17: return K::F17;
    // case SDLK_F18: return K::F18;
    // case SDLK_F19: return K::F19;
    // case SDLK_F20: return K::F20;
    // case SDLK_F21: return K::F21;
    // case SDLK_F22: return K::F22;
    // case SDLK_F23: return K::F23;
    // case SDLK_F24: return K::F24;
    // case SDLK_AC_BACK: return K::APP_BACK;
    // case SDLK_AC_FORWARD: return K::APPFORWARD;
    default:
        break;
    }

    // Fallback to scancode
    switch (scancode) {
    case SDL_SCANCODE_GRAVE:
        return K::GRAVE_ACCENT;
    case SDL_SCANCODE_MINUS:
        return K::MINUS;
    case SDL_SCANCODE_EQUALS:
        return K::EQUAL;
    case SDL_SCANCODE_LEFTBRACKET:
        return K::LEFT_BRACKET;
    case SDL_SCANCODE_RIGHTBRACKET:
        return K::RIGHT_BRACKET;
    // case SDL_SCANCODE_NONUSBACKSLASH: return K::OEM102;
    case SDL_SCANCODE_BACKSLASH:
        return K::BACKSLASH;
    case SDL_SCANCODE_SEMICOLON:
        return K::SEMICOLON;
    case SDL_SCANCODE_APOSTROPHE:
        return K::APOSTROPHE;
    case SDL_SCANCODE_COMMA:
        return K::COMMA;
    case SDL_SCANCODE_PERIOD:
        return K::PERIOD;
    case SDL_SCANCODE_SLASH:
        return K::SLASH;
    default:
        break;
    }
    return K::UNKNOWN;
}

InputController::KeyStatus status_from_sdl(const bool down) {
    return down ? InputController::KeyStatus::PRESS : InputController::KeyStatus::RELEASE;
}

int mods_from_sdl(const SDL_Keymod sdl_mods) {
    int mods = 0;
    mods |= (sdl_mods & SDL_KMOD_SHIFT) ? InputController::SHIFT : 0;
    mods |= (sdl_mods & SDL_KMOD_CTRL) ? InputController::CONTROL : 0;
    mods |= (sdl_mods & SDL_KMOD_ALT) ? InputController::ALT : 0;
    mods |= (sdl_mods & SDL_KMOD_GUI) ? InputController::SUPER : 0;
    mods |= (sdl_mods & SDL_KMOD_CAPS) ? InputController::CAPS_LOCK : 0;
    mods |= (sdl_mods & SDL_KMOD_NUM) ? InputController::NUM_LOCK : 0;
    return mods;
}

// clang-format off
InputController::MouseButton mouse_button_from_sdl(const Uint8 sdl_button) {
    switch (sdl_button) {
    case SDL_BUTTON_LEFT:   return InputController::MouseButton::MOUSE1;
    case SDL_BUTTON_RIGHT:  return InputController::MouseButton::MOUSE2;
    case SDL_BUTTON_MIDDLE: return InputController::MouseButton::MOUSE3;
    case SDL_BUTTON_X1:     return InputController::MouseButton::MOUSE4;
    case SDL_BUTTON_X2:     return InputController::MouseButton::MOUSE5;
    default:                return InputController::MouseButton::UNKNOWN;
    }
}
// clang-format on

} // namespace

SDLWindow::SDLWindow(const DeviceHandle& device,
                     const int width,
                     const int height,
                     const char* title)
    : device(device) {
    SPDLOG_DEBUG("create SDL window ({})", fmt::ptr(this));
    window =
        SDL_CreateWindow(title, width, height,
                         SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (window == nullptr)
        throw std::runtime_error(fmt::format("SDL_CreateWindow failed: {}", SDL_GetError()));

    SDL_SetPointerProperty(SDL_GetWindowProperties(window), SDL_WINDOW_USER_DATA_KEY, this);

    {
        int ww, wh, pw, ph;
        SDL_GetWindowSize(window, &ww, &wh);
        SDL_GetWindowSizeInPixels(window, &pw, &ph);
        SPDLOG_INFO("SDL window created: driver={}, window={}x{}, pixels={}x{}, "
                    "content_scale={:.2f}, display_scale={:.2f}",
                    SDL_GetCurrentVideoDriver() ? SDL_GetCurrentVideoDriver() : "?", ww, wh, pw, ph,
                    SDL_GetDisplayContentScale(SDL_GetDisplayForWindow(window)),
                    SDL_GetWindowDisplayScale(window));
    }

    // clang-format off
    cursors[static_cast<size_t>(WindowCursorShape::Arrow)]      = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_DEFAULT);
    cursors[static_cast<size_t>(WindowCursorShape::TextInput)]  = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_TEXT);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeAll)]  = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_MOVE);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeNS)]   = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NS_RESIZE);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeEW)]   = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_EW_RESIZE);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeNESW)] = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NESW_RESIZE);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeNWSE)] = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NWSE_RESIZE);
    cursors[static_cast<size_t>(WindowCursorShape::Hand)]       = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_POINTER);
    cursors[static_cast<size_t>(WindowCursorShape::NotAllowed)] = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NOT_ALLOWED);
    // Hidden and Count slots remain nullptr
    // clang-format on
}

SDLWindow::~SDLWindow() {
    SPDLOG_DEBUG("destroy SDL window ({})", fmt::ptr(this));
    for (SDL_Cursor* c : cursors)
        if (c != nullptr)
            SDL_DestroyCursor(c);
    if (clipboard != nullptr) {
        SDL_free(clipboard);
    }
    SDL_DestroyWindow(window);
}

void SDLWindow::dispatch_event(const SDL_Event& event) {
    SDL_WindowID window_id = 0;
    switch (event.type) {
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
    case SDL_EVENT_WINDOW_RESIZED:
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
    case SDL_EVENT_WINDOW_FOCUS_LOST:
    case SDL_EVENT_WINDOW_MINIMIZED:
    case SDL_EVENT_WINDOW_RESTORED:
        window_id = event.window.windowID;
        break;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        window_id = event.key.windowID;
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        window_id = event.button.windowID;
        break;
    case SDL_EVENT_MOUSE_MOTION:
        window_id = event.motion.windowID;
        break;
    case SDL_EVENT_MOUSE_WHEEL:
        window_id = event.wheel.windowID;
        break;
    case SDL_EVENT_TEXT_INPUT:
        window_id = event.text.windowID;
        break;
    default:
        return;
    }

    SDL_Window* sdl_win = SDL_GetWindowFromID(window_id);
    if (sdl_win == nullptr)
        return;
    auto* self = static_cast<SDLWindow*>(SDL_GetPointerProperty(SDL_GetWindowProperties(sdl_win),
                                                                SDL_WINDOW_USER_DATA_KEY, nullptr));
    if (self != nullptr)
        self->handle_event(event);
}

void SDLWindow::handle_event(const SDL_Event& event) {
    switch (event.type) {
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
        close_requested = true;
        if (close_cb)
            close_cb();
        break;
    case SDL_EVENT_MOUSE_MOTION:
        if (SDL_GetWindowRelativeMouseMode(window)) {
            rel_accum_x += event.motion.xrel;
            rel_accum_y += event.motion.yrel;
            dispatch_cursor(rel_accum_x, rel_accum_y);
        } else {
            dispatch_cursor(event.motion.x, event.motion.y);
        }
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        dispatch_mouse_button(mouse_button_from_sdl(event.button.button),
                              status_from_sdl(event.button.down));
        break;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        dispatch_key(key_from_sdl(event.key.key, event.key.scancode),
                     status_from_sdl(event.key.down), mods_from_sdl(event.key.mod));
        break;
    }
    case SDL_EVENT_MOUSE_WHEEL:
        dispatch_scroll(-event.wheel.x, event.wheel.y);
        break;
    case SDL_EVENT_TEXT_INPUT: {
        const char* text = event.text.text;
        size_t len = SDL_strlen(text);
        while (len > 0) {
            const Uint32 cp = SDL_StepUTF8(&text, &len);
            if (cp != 0u)
                dispatch_char(cp);
        }
        break;
    }
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: {
        int ww, wh;
        SDL_GetWindowSize(window, &ww, &wh);
        dispatch_resize(
            {static_cast<uint32_t>(event.window.data1), static_cast<uint32_t>(event.window.data2)},
            {static_cast<uint32_t>(ww), static_cast<uint32_t>(wh)});
        break;
    }
    case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
        dispatch_display_scale_changed(SDL_GetWindowDisplayScale(window));
        break;
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
        dispatch_focus_changed(true);
        break;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        dispatch_focus_changed(false);
        break;
    case SDL_EVENT_WINDOW_MINIMIZED:
        dispatch_minimized();
        break;
    case SDL_EVENT_WINDOW_RESTORED:
        dispatch_restored();
        break;
    default:
        break;
    }
}

vk::Extent2D SDLWindow::framebuffer_extent() {
    int w, h;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    return vk::Extent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
}

vk::Extent2D SDLWindow::window_extent() {
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    return vk::Extent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
}

SurfaceHandle SDLWindow::get_surface() {
    return std::make_shared<SDLSurface>(device,
                                        std::static_pointer_cast<SDLWindow>(shared_from_this()));
}

bool SDLWindow::should_close() const {
    return close_requested;
}

void SDLWindow::poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event))
        dispatch_event(event);
}

bool SDLWindow::is_fullscreen() const {
    return (SDL_GetWindowFlags(window) & SDL_WINDOW_FULLSCREEN) != 0;
}

void SDLWindow::set_fullscreen(const bool fullscreen) {
    SDL_SetWindowFullscreen(window, fullscreen);
}

void SDLWindow::set_size(const uint32_t width, const uint32_t height) {
    SDL_SetWindowSize(window, static_cast<int>(width), static_cast<int>(height));
}

void SDLWindow::set_title(const std::string& title) {
    SDL_SetWindowTitle(window, title.c_str());
}

InputControllerHandle SDLWindow::get_input_controller() {
    return std::static_pointer_cast<InputController>(
        std::static_pointer_cast<SDLWindow>(shared_from_this()));
}

void SDLWindow::set_close_callback(const CloseCallback& cb) {
    close_cb = cb;
}

SDL_Window* SDLWindow::get_window() const {
    return window;
}

void SDLWindow::set_cursor_pos(const double x, const double y) {
    SDL_WarpMouseInWindow(window, static_cast<float>(x), static_cast<float>(y));
}

void SDLWindow::set_cursor(const WindowCursorShape shape) {
    if (shape == current_cursor_shape)
        return;
    current_cursor_shape = shape;

    if (is_mouse_grabbed())
        return;

    if (shape == WindowCursorShape::Hidden) {
        SDL_HideCursor();
    } else {
        SDL_SetCursor(cursors[static_cast<size_t>(shape)]);
        SDL_ShowCursor();
    }
}

bool SDLWindow::set_mouse_grabbed(const bool grabbed) {
    if (!SDL_GetWindowRelativeMouseMode(window)) {
        SDL_GetMouseState(&saved_cursor_x, &saved_cursor_y);
        rel_accum_x = 0;
        rel_accum_y = 0;
    }
    if (!grabbed)
        SDL_WarpMouseInWindow(window, saved_cursor_x, saved_cursor_y);
    SDL_SetWindowRelativeMouseMode(window, grabbed);
    bool success = SDL_SetWindowMouseGrab(window, grabbed);

    if (!grabbed) {
        if (current_cursor_shape == WindowCursorShape::Hidden) {
            SDL_HideCursor();
        } else {
            SDL_SetCursor(cursors[static_cast<size_t>(current_cursor_shape)]);
            SDL_ShowCursor();
        }
    }

    return success;
}

float SDLWindow::get_display_scale() {
    return SDL_GetWindowDisplayScale(window);
}

bool SDLWindow::is_mouse_grabbed() const {
    return SDL_GetWindowMouseGrab(window);
}

void SDLWindow::set_clipboard_text(const char* text) {
    SDL_SetClipboardText(text);
}

const char* SDLWindow::get_clipboard_text() {
    if (clipboard != nullptr) {
        SDL_free(clipboard);
    }
    clipboard = SDL_GetClipboardText();
    return clipboard;
}

void SDLWindow::start_text_input() {
    SDL_StartTextInput(window);
}

void SDLWindow::stop_text_input() {
    SDL_StopTextInput(window);
}

bool SDLWindow::is_text_input_active() const {
    return SDL_TextInputActive(window);
}

void SDLWindow::set_text_input_area(const int x, const int y, const int w, const int h) {
    SDL_Rect r{x, y, w, h};
    SDL_SetTextInputArea(window, &r, 0);
}

} // namespace merian
