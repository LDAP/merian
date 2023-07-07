#pragma once

#include "merian/utils/input_controller.hpp"
#include "merian/vk/window/glfw_window.hpp"

namespace merian {

class GLFWInputController : public InputController {

private:
    friend void glfw_cursor_cb(GLFWwindow* window, double xpos, double ypos);
    friend void glfw_mouseb_cb(GLFWwindow* window, int glfw_button, int action, int glfw_mods);
    friend void glfw_key_cb(GLFWwindow* window, int key, int scancode, int action, int glfw_mods);
    friend void glfw_scroll_cb(GLFWwindow* window, double xoffset, double yoffset);

  public:
    explicit GLFWInputController(const GLFWWindowHandle window);

    ~GLFWInputController();

    virtual bool request_raw_mouse_input(bool enable) override;

    // Returns true if raw mouse input is enabled.
    virtual bool get_raw_mouse_input() override;

    // Clear all callbacks
    virtual void reset() override;

    virtual void set_active(bool active) override;

    virtual void set_mouse_cursor_callback(MouseCursorEventCallback cb) override;
    virtual void set_mouse_button_callback(MouseButtonEventCallback cb) override;
    virtual void set_scroll_event_callback(ScrollEventCallback cb) override;
    virtual void set_key_event_callback(KeyEventCallback cb) override;

  private:
    const GLFWWindowHandle window;

    MouseCursorEventCallback cursor_cb{nullptr};
    MouseButtonEventCallback mbutton_cb{nullptr};
    KeyEventCallback key_cb{nullptr};
    ScrollEventCallback scroll_cb{nullptr};

    bool active = true;
};

} // namespace merian
