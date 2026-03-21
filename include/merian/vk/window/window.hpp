#pragma once

#include "merian/vk/window/surface.hpp"
#include "merian/vk/window/window_listener.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace merian {

enum class WindowPlatform { GLFW, SDL, Other };

// Platform-agnostic cursor shape (mirrors ImGuiMouseCursor without pulling in imgui.h).
enum class WindowCursorShape {
    Arrow,
    TextInput,
    ResizeAll,
    ResizeNS,
    ResizeEW,
    ResizeNESW,
    ResizeNWSE,
    Hand,
    NotAllowed,
    Hidden, // hidden but free to move
    Count,  // sentinel — must be last; use for array sizing only
};

struct WindowCreateInfo {
    int width = 1280;
    int height = 720;
    std::string title = "merian";
};

class InputController;
using InputControllerHandle = std::shared_ptr<InputController>;

/*
 * Abstract window interface. Implemented by GLFWWindow and SDLWindow.
 *
 * poll_events() drains the platform event queue and dispatches all callbacks
 * (both window-level and input).
 *
 * Input events (key, mouse, cursor, scroll) are delivered through the InputController
 * returned by get_input_controller().
 */
class Window : public std::enable_shared_from_this<Window> {
  public:
    using CloseCallback = std::function<void()>;

    virtual ~Window() = default;

    // --- Core interface ---

    // Size of the framebuffer in pixels.
    virtual vk::Extent2D framebuffer_extent() = 0;

    // Size of the window in screen coordinates (before DPI scaling).
    // On non-HiDPI displays this equals framebuffer_extent().
    virtual vk::Extent2D window_extent() = 0;

    virtual SurfaceHandle get_surface() = 0;

    // Returns true if the user has requested to close this window.
    virtual bool should_close() const = 0;

    // Process all pending platform events and dispatch registered callbacks.
    virtual void poll_events() = 0;

    virtual bool is_fullscreen() const = 0;

    // Enter or leave fullscreen. Implementations are responsible for saving and
    // restoring the windowed position/size internally.
    virtual void set_fullscreen(const bool fullscreen) = 0;

    // Resize the window to the given pixel dimensions (no-op in fullscreen).
    virtual void set_size(const uint32_t width, const uint32_t height) = 0;

    // Update the window title.
    virtual void set_title(const std::string& title) = 0;

    // Returns (or lazily creates) the InputController for this window.
    // All key/mouse/cursor/scroll events are routed through it.
    virtual InputControllerHandle get_input_controller() = 0;

    // Ratio of framebuffer pixels to window screen coordinates (logical pixels).
    virtual float get_pixel_density() {
        const auto fb = framebuffer_extent();
        const auto win = window_extent();
        return win.width > 0 ? static_cast<float>(fb.width) / static_cast<float>(win.width) : 1.0f;
    }

    // This is a combination of the window pixel density and the display content scale (the OS
    // setting), and is the expected scale for displaying content in this window. For example, if a
    // 3840x2160 window had a display scale of 2.0, the user expects the content to take twice as
    // many pixels and be the same physical size as if it were being displayed in a 1920x1080 window
    // with a display scale of 1.0.
    virtual float get_display_scale() {
        return get_pixel_density();
    }

    // Called when the user requests to close the window (OS close button etc.)
    virtual void set_close_callback(const CloseCallback& /*cb*/) {}

    virtual WindowPlatform get_platform_type() const {
        return WindowPlatform::Other;
    }

    // --- Platform services (default no-ops) ---

    virtual void set_cursor_pos(double /*x*/, double /*y*/) {}
    virtual void set_cursor(WindowCursorShape /*shape*/) {}
    // Grab/ungrab mouse: hide cursor, enable raw/relative input, confine to window.
    // Returns true if supported. When ungrabbed, restores the last set_cursor() shape.
    virtual bool set_mouse_grabbed(bool /*grabbed*/) {
        return false;
    }
    virtual bool is_mouse_grabbed() const {
        return false;
    }
    virtual void set_clipboard_text(const char* /*text*/) {}
    // returns nullptr if clipboard is not set or not supported.
    virtual const char* get_clipboard_text() {
        return nullptr;
    }

    // --- Window listener dispatch ---

    void add_window_listener(std::weak_ptr<WindowListener> listener);
    void clear_window_listeners();

  protected:
    void dispatch_resize(vk::Extent2D framebuffer_extent, vk::Extent2D window_extent);
    void dispatch_display_scale_changed(float display_scale);
    void dispatch_focus_changed(bool focused);
    void dispatch_minimized();
    void dispatch_restored();

  private:
    std::vector<std::weak_ptr<WindowListener>> window_listeners;
};

using WindowHandle = std::shared_ptr<Window>;

} // namespace merian
