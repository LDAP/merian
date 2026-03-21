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
    vk::Extent2D window_extent() override;
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

    WindowPlatform get_platform_type() const override {
        return WindowPlatform::GLFW;
    }

    // --- Platform services ---

    void set_cursor_pos(double x, double y) override;
    void set_cursor(WindowCursorShape shape) override;
    bool set_mouse_grabbed(bool grabbed) override;
    bool is_mouse_grabbed() const override;
    void set_clipboard_text(const char* text) override;
    const char* get_clipboard_text() override;
    float get_display_scale() override;

    // --- GLFW-specific ---

    operator GLFWwindow*() const;
    GLFWwindow* get_window() const;

  private:
    // Static GLFW callback handlers.
    static void cb_close(GLFWwindow* w);
    static void cb_cursor(GLFWwindow* w, double xpos, double ypos);
    static void cb_mouse_button(GLFWwindow* w, int button, int action, int mods);
    static void cb_key(GLFWwindow* w, int key, int scancode, int action, int mods);
    static void cb_scroll(GLFWwindow* w, double xoffset, double yoffset);
    static void cb_char(GLFWwindow* w, unsigned int codepoint);
    static void cb_framebuffer_size(GLFWwindow* w, int width, int height);
    static void cb_content_scale(GLFWwindow* w, float xscale, float yscale);
    static void cb_focus(GLFWwindow* w, int focused);
    static void cb_iconify(GLFWwindow* w, int iconified);

    DeviceHandle device;
    GLFWwindow* window = nullptr;

    // saved when entering fullscreen, restored on exit
    std::array<int, 4> windowed_pos_size{100, 100, 1280, 720};

    // Pre-created system cursors indexed by WindowCursorShape; nullptr for Hidden and Count.
    std::array<GLFWcursor*, static_cast<size_t>(WindowCursorShape::Count)> cursors{};
    WindowCursorShape current_cursor_shape = WindowCursorShape::Arrow;

    CloseCallback close_cb;
};

using GLFWWindowHandle = std::shared_ptr<GLFWWindow>;

} // namespace merian
