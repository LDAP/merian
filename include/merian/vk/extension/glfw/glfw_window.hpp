#pragma once

#include "merian/utils/input_controller.hpp"
#include "merian/vk/window/window.hpp"

#include <GLFW/glfw3.h>
#include <array>
#include <spdlog/spdlog.h>
#include <string>

namespace merian {

class ExtensionGLFW;

class GLFWWindow : public Window, public InputController {
  public:
    friend class ExtensionGLFW;

  private:
    GLFWWindow(const DeviceHandle& device, const WindowCreateInfo& create_info = {});

  public:
    ~GLFWWindow() override;

    // --- Window interface ---

    vk::Extent2D framebuffer_extent() override;
    SurfaceHandle get_surface() override;

    bool should_close() const override;
    void poll_events() override;

    bool is_fullscreen() const override;
    // Saves windowed pos/size on enter; restores on exit.
    // Silently ignores GLFW_FEATURE_UNAVAILABLE (Wayland position query).
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

    // --- GLFW-specific ---

    operator GLFWwindow*() const;
    GLFWwindow* get_window() const;

  private:
    // Static GLFW callback handlers — valid C function pointers, access private
    // members via glfwGetWindowUserPointer (the standard GLFW pattern for state).
    static void cb_close(GLFWwindow* w);
    static void cb_cursor(GLFWwindow* w, double xpos, double ypos);
    static void cb_mouse_button(GLFWwindow* w, int button, int action, int mods);
    static void cb_key(GLFWwindow* w, int key, int scancode, int action, int mods);
    static void cb_scroll(GLFWwindow* w, double xoffset, double yoffset);

    DeviceHandle device;
    GLFWwindow* window = nullptr;

    // saved when entering fullscreen, restored on exit
    std::array<int, 4> windowed_pos_size{100, 100, 1280, 720};

    CloseCallback close_cb;

    // InputController state
    MouseCursorEventCallback cursor_cb{nullptr};
    MouseButtonEventCallback mbutton_cb{nullptr};
    KeyEventCallback key_cb{nullptr};
    ScrollEventCallback scroll_cb{nullptr};
    bool input_active = true;

    // Previously registered GLFW callbacks, saved for chaining.
    GLFWcursorposfun prev_cursor_cb{nullptr};
    GLFWmousebuttonfun prev_mbutton_cb{nullptr};
    GLFWkeyfun prev_key_cb{nullptr};
    GLFWscrollfun prev_scroll_cb{nullptr};
};

using GLFWWindowHandle = std::shared_ptr<GLFWWindow>;

} // namespace merian
