#pragma once

#include "merian/utils/input_controller.hpp"
#include "merian/vk/window/window.hpp"

#include <cstdint>
#include <string>

// Forward declarations to avoid including SDL headers from this header
struct SDL_Window;
union SDL_Event;

namespace merian {

class ExtensionSDLWindow;

class SDLWindow : public Window, public InputController {
  public:
    friend class ExtensionSDLWindow;

  private:
    SDLWindow(const DeviceHandle& device,
              const int width = 1280,
              const int height = 720,
              const char* title = "");

  public:
    ~SDLWindow() override;

    // --- Window interface ---

    vk::Extent2D framebuffer_extent() override;
    SurfaceHandle get_surface() override;

    bool should_close() const override;
    void poll_events() override;

    bool is_fullscreen() const override;
    void set_fullscreen(const bool fullscreen) override;

    void set_size(const uint32_t width, const uint32_t height) override;
    void set_title(const std::string& title) override;

    InputControllerHandle get_input_controller() override;

    void set_close_callback(const CloseCallback& cb) override;

    // --- InputController interface ---

    bool request_raw_mouse_input(bool enable) override;
    bool get_raw_mouse_input() override;
    void reset() override;
    void set_active(bool active) override;
    void set_mouse_cursor_callback(const MouseCursorEventCallback& cb) override;
    void set_mouse_button_callback(const MouseButtonEventCallback& cb) override;
    void set_scroll_event_callback(const ScrollEventCallback& cb) override;
    void set_key_event_callback(const KeyEventCallback& cb) override;

    // --- SDL-specific ---

    SDL_Window* get_window() const;

    // Routes an SDL event to the SDLWindow instance stored as window user data.
    static void dispatch_event(const SDL_Event& event);

  private:
    void handle_event(const SDL_Event& event);

    DeviceHandle device;
    SDL_Window* window = nullptr;
    bool close_requested = false;

    CloseCallback close_cb;

    // InputController state
    MouseCursorEventCallback cursor_cb{nullptr};
    MouseButtonEventCallback mbutton_cb{nullptr};
    KeyEventCallback key_cb{nullptr};
    ScrollEventCallback scroll_cb{nullptr};
    bool input_active = true;
    bool raw_mouse = false;
};

using SDLWindowHandle = std::shared_ptr<SDLWindow>;

} // namespace merian
