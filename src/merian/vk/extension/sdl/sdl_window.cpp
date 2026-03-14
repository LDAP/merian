#include "merian/vk/extension/sdl/sdl_window.hpp"
#include "merian/vk/extension/sdl/sdl_surface.hpp"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

namespace merian {

namespace {

static constexpr const char* SDL_WINDOW_USER_DATA_KEY = "merian_sdl_window";

InputController::KeyStatus status_from_sdl(const bool down) {
    return down ? InputController::PRESS : InputController::RELEASE;
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

InputController::MouseButton mouse_button_from_sdl(const Uint8 sdl_button) {
    switch (sdl_button) {
    case SDL_BUTTON_LEFT:
        return InputController::MOUSE1;
    case SDL_BUTTON_RIGHT:
        return InputController::MOUSE2;
    case SDL_BUTTON_MIDDLE:
        return InputController::MOUSE3;
    case SDL_BUTTON_X1:
        return InputController::MOUSE4;
    case SDL_BUTTON_X2:
        return InputController::MOUSE5;
    default:
        throw std::invalid_argument{"unsupported SDL mouse button"};
    }
}

} // namespace

SDLWindow::SDLWindow(const DeviceHandle& device,
                     const int width,
                     const int height,
                     const char* title)
    : device(device) {
    SPDLOG_DEBUG("create SDL window ({})", fmt::ptr(this));
    window = SDL_CreateWindow(title, width, height,
                              SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!window)
        throw std::runtime_error(fmt::format("SDL_CreateWindow failed: {}", SDL_GetError()));

    SDL_SetPointerProperty(SDL_GetWindowProperties(window), SDL_WINDOW_USER_DATA_KEY, this);
}

SDLWindow::~SDLWindow() {
    SPDLOG_DEBUG("destroy SDL window ({})", fmt::ptr(this));
    SDL_DestroyWindow(window);
}

void SDLWindow::dispatch_event(const SDL_Event& event) {
    SDL_WindowID window_id = 0;
    switch (event.type) {
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
    case SDL_EVENT_WINDOW_RESIZED:
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
    default:
        return;
    }

    SDL_Window* sdl_win = SDL_GetWindowFromID(window_id);
    if (!sdl_win)
        return;
    auto* self = static_cast<SDLWindow*>(
        SDL_GetPointerProperty(SDL_GetWindowProperties(sdl_win), SDL_WINDOW_USER_DATA_KEY, nullptr));
    if (self)
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
        if (input_active && cursor_cb)
            cursor_cb(*this, event.motion.x, event.motion.y);
        break;
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (input_active && mbutton_cb) {
            try {
                mbutton_cb(*this, mouse_button_from_sdl(event.button.button),
                           status_from_sdl(event.button.down), 0);
            } catch (const std::invalid_argument&) {
            }
        }
        break;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        if (input_active && key_cb) {
            const int mods = mods_from_sdl(event.key.mod);
            key_cb(*this, event.key.key, event.key.scancode,
                   status_from_sdl(event.key.down), mods);
        }
        break;
    case SDL_EVENT_MOUSE_WHEEL:
        if (input_active && scroll_cb)
            scroll_cb(*this, event.wheel.x, event.wheel.y);
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

bool SDLWindow::request_raw_mouse_input(const bool enable) {
    SDL_SetWindowRelativeMouseMode(window, enable);
    raw_mouse = enable;
    return true;
}

bool SDLWindow::get_raw_mouse_input() {
    return raw_mouse;
}

void SDLWindow::reset() {
    cursor_cb  = nullptr;
    mbutton_cb = nullptr;
    key_cb     = nullptr;
    scroll_cb  = nullptr;
}

void SDLWindow::set_active(const bool active) {
    input_active = active;
}

void SDLWindow::set_mouse_cursor_callback(const MouseCursorEventCallback& cb) {
    cursor_cb = cb;
}

void SDLWindow::set_mouse_button_callback(const MouseButtonEventCallback& cb) {
    mbutton_cb = cb;
}

void SDLWindow::set_scroll_event_callback(const ScrollEventCallback& cb) {
    scroll_cb = cb;
}

void SDLWindow::set_key_event_callback(const KeyEventCallback& cb) {
    key_cb = cb;
}

SDL_Window* SDLWindow::get_window() const {
    return window;
}

} // namespace merian
