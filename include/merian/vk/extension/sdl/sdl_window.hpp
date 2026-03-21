#pragma once

#include "merian/utils/input_controller.hpp"
#include "merian/vk/window/window.hpp"

#include <cstdint>
#include <string>

// Forward declarations to avoid including SDL headers from this header
struct SDL_Window;
struct SDL_Cursor;
union SDL_Event;

namespace merian {

class ExtensionSDLVideo;

class SDLWindow : public Window, public InputController {
  public:
    friend class ExtensionSDLVideo;

  private:
    SDLWindow(const DeviceHandle& device,
              const int width = 1280,
              const int height = 720,
              const char* title = "");

  public:
    ~SDLWindow() override;

    // --- Window interface ---

    vk::Extent2D framebuffer_extent() override;
    vk::Extent2D window_extent() override;
    SurfaceHandle get_surface() override;

    bool should_close() const override;
    void poll_events() override;

    bool is_fullscreen() const override;
    void set_fullscreen(const bool fullscreen) override;

    void set_size(const uint32_t width, const uint32_t height) override;
    void set_title(const std::string& title) override;

    InputControllerHandle get_input_controller() override;

    void set_close_callback(const CloseCallback& cb) override;

    WindowPlatform get_platform_type() const override {
        return WindowPlatform::SDL;
    }

    // --- Platform services ---

    void set_cursor_pos(double x, double y) override;
    void set_cursor(WindowCursorShape shape) override;
    bool set_mouse_grabbed(bool grabbed) override;
    bool is_mouse_grabbed() const override;
    void set_clipboard_text(const char* text) override;
    const char* get_clipboard_text() override;
    float get_display_scale() override;

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
    char* clipboard = nullptr;

    // Pre-created system cursors indexed by WindowCursorShape; nullptr for Hidden and Count.
    std::array<SDL_Cursor*, static_cast<size_t>(WindowCursorShape::Count)> cursors{};
    WindowCursorShape current_cursor_shape = WindowCursorShape::Arrow;

    // Accumulated relative motion — reset when entering relative mode.
    double rel_accum_x = 0;
    double rel_accum_y = 0;

    // Cursor position saved on grab; restored on ungrab.
    float saved_cursor_x = 0;
    float saved_cursor_y = 0;
};

using SDLWindowHandle = std::shared_ptr<SDLWindow>;

} // namespace merian
