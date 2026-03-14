#pragma once

#include "merian/vk/window/surface.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vulkan/vulkan.hpp>

namespace merian {

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
 * (both window-level and input). It is safe to call from multiple WindowNodes:
 *   - GLFW: glfwPollEvents() is idempotent.
 *   - SDL:  a static dispatcher routes each event to the correct window instance.
 *
 * Input events (key, mouse, cursor, scroll) are delivered through the InputController
 * returned by get_input_controller().
 */
class Window : public std::enable_shared_from_this<Window> {
  public:
    using CloseCallback = std::function<void()>;

    virtual ~Window() = default;

    // --- Core interface ---

    virtual vk::Extent2D framebuffer_extent() = 0;

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

    // Called when the user requests to close the window (OS close button etc.)
    virtual void set_close_callback(const CloseCallback& /*cb*/) {}
};

using WindowHandle = std::shared_ptr<Window>;

} // namespace merian
