#include "merian/vk/extension/glfw/glfw_window.hpp"
#include "merian/vk/extension/glfw/extension_glfw.hpp"
#include "merian/vk/extension/glfw/glfw_surface.hpp"

#include <spdlog/spdlog.h>

namespace merian {

namespace {

int mods_from_glfw(const int glfw_mods) {
    int mods = 0;
    mods |= (glfw_mods & GLFW_MOD_SHIFT) ? InputController::SHIFT : 0;
    mods |= (glfw_mods & GLFW_MOD_CONTROL) ? InputController::CONTROL : 0;
    mods |= (glfw_mods & GLFW_MOD_ALT) ? InputController::ALT : 0;
    mods |= (glfw_mods & GLFW_MOD_SUPER) ? InputController::SUPER : 0;
    mods |= (glfw_mods & GLFW_MOD_CAPS_LOCK) ? InputController::CAPS_LOCK : 0;
    mods |= (glfw_mods & GLFW_MOD_NUM_LOCK) ? InputController::NUM_LOCK : 0;
    return mods;
}

InputController::KeyStatus status_from_glfw(const int action) {
    switch (action) {
    case GLFW_PRESS:
        return InputController::PRESS;
    case GLFW_RELEASE:
        return InputController::RELEASE;
    case GLFW_REPEAT:
        return InputController::REPEAT;
    default:
        throw std::invalid_argument{"unsupported GLFW action"};
    }
}

InputController::MouseButton mouse_button_from_glfw(const int glfw_button) {
    switch (glfw_button) {
    case GLFW_MOUSE_BUTTON_1:
        return InputController::MOUSE1;
    case GLFW_MOUSE_BUTTON_2:
        return InputController::MOUSE2;
    case GLFW_MOUSE_BUTTON_3:
        return InputController::MOUSE3;
    case GLFW_MOUSE_BUTTON_4:
        return InputController::MOUSE4;
    case GLFW_MOUSE_BUTTON_5:
        return InputController::MOUSE5;
    default:
        throw std::invalid_argument{"unsupported GLFW mouse button"};
    }
}

} // namespace

void GLFWWindow::cb_close(GLFWwindow* w) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    if (self->close_cb)
        self->close_cb();
}

void GLFWWindow::cb_cursor(GLFWwindow* w, const double xpos, const double ypos) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    if (self->input_active && self->cursor_cb)
        self->cursor_cb(*self, xpos, ypos);
    if (self->prev_cursor_cb)
        self->prev_cursor_cb(w, xpos, ypos);
}

void GLFWWindow::cb_mouse_button(GLFWwindow* w,
                                  const int glfw_button,
                                  const int action,
                                  const int glfw_mods) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    if (self->input_active && self->mbutton_cb) {
        try {
            self->mbutton_cb(*self, mouse_button_from_glfw(glfw_button),
                             status_from_glfw(action), mods_from_glfw(glfw_mods));
        } catch (const std::invalid_argument&) {
        }
    }
    if (self->prev_mbutton_cb)
        self->prev_mbutton_cb(w, glfw_button, action, glfw_mods);
}

void GLFWWindow::cb_key(GLFWwindow* w,
                         const int key,
                         const int scancode,
                         const int action,
                         const int glfw_mods) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    if (self->input_active && self->key_cb) {
        try {
            self->key_cb(*self, key, scancode, status_from_glfw(action),
                         mods_from_glfw(glfw_mods));
        } catch (const std::invalid_argument&) {
        }
    }
    if (self->prev_key_cb)
        self->prev_key_cb(w, key, scancode, action, glfw_mods);
}

void GLFWWindow::cb_scroll(GLFWwindow* w, const double xoffset, const double yoffset) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    if (self->input_active && self->scroll_cb)
        self->scroll_cb(*self, xoffset, yoffset);
    if (self->prev_scroll_cb)
        self->prev_scroll_cb(w, xoffset, yoffset);
}

GLFWWindow::GLFWWindow(const DeviceHandle& device,
                       const int width,
                       const int height,
                       const char* title)
    : device(device) {
    SPDLOG_DEBUG("create window ({})", fmt::ptr(this));
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);

    glfwSetWindowUserPointer(window, this);
    glfwSetWindowCloseCallback(window, cb_close);
    prev_cursor_cb  = glfwSetCursorPosCallback(window, cb_cursor);
    prev_mbutton_cb = glfwSetMouseButtonCallback(window, cb_mouse_button);
    prev_key_cb     = glfwSetKeyCallback(window, cb_key);
    prev_scroll_cb  = glfwSetScrollCallback(window, cb_scroll);
}

GLFWWindow::~GLFWWindow() {
    SPDLOG_DEBUG("destroy window ({})", fmt::ptr(this));
    glfwDestroyWindow(window);
}

vk::Extent2D GLFWWindow::framebuffer_extent() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    return vk::Extent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

SurfaceHandle GLFWWindow::get_surface() {
    return std::make_shared<GLFWSurface>(device,
                                         std::static_pointer_cast<GLFWWindow>(shared_from_this()));
}

bool GLFWWindow::should_close() const {
    return glfwWindowShouldClose(window) == GLFW_TRUE;
}

void GLFWWindow::poll_events() {
    glfwPollEvents();
}

bool GLFWWindow::is_fullscreen() const {
    return glfwGetWindowMonitor(window) != nullptr;
}

void GLFWWindow::set_fullscreen(const bool fullscreen) {
    if (fullscreen == is_fullscreen())
        return;

    if (fullscreen) {
        try {
            glfwGetWindowPos(window, &windowed_pos_size[0], &windowed_pos_size[1]);
        } catch (const ExtensionGLFW::glfw_error& e) {
            if (e.id != GLFW_FEATURE_UNAVAILABLE)
                throw;
            windowed_pos_size[0] = windowed_pos_size[1] = 0;
        }
        glfwGetWindowSize(window, &windowed_pos_size[2], &windowed_pos_size[3]);

        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* vidmode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window, monitor, 0, 0, vidmode->width, vidmode->height,
                             vidmode->refreshRate);
    } else {
        glfwSetWindowMonitor(window, nullptr, windowed_pos_size[0], windowed_pos_size[1],
                             windowed_pos_size[2], windowed_pos_size[3], GLFW_DONT_CARE);
    }
}

void GLFWWindow::set_size(const uint32_t width, const uint32_t height) {
    glfwSetWindowSize(window, static_cast<int>(width), static_cast<int>(height));
}

void GLFWWindow::set_title(const std::string& title) {
    glfwSetWindowTitle(window, title.c_str());
}

InputControllerHandle GLFWWindow::get_input_controller() {
    return std::static_pointer_cast<InputController>(
        std::static_pointer_cast<GLFWWindow>(shared_from_this()));
}

void GLFWWindow::set_close_callback(const CloseCallback& cb) {
    close_cb = cb;
}

bool GLFWWindow::request_raw_mouse_input(const bool enable) {
    if (glfwRawMouseMotionSupported() == 0)
        return false;

    if (enable) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    } else {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
    return true;
}

bool GLFWWindow::get_raw_mouse_input() {
    return glfwGetInputMode(window, GLFW_RAW_MOUSE_MOTION) != 0;
}

void GLFWWindow::reset() {
    cursor_cb  = nullptr;
    mbutton_cb = nullptr;
    key_cb     = nullptr;
    scroll_cb  = nullptr;
}

void GLFWWindow::set_active(const bool active) {
    input_active = active;
}

void GLFWWindow::set_mouse_cursor_callback(const MouseCursorEventCallback& cb) {
    cursor_cb = cb;
}

void GLFWWindow::set_mouse_button_callback(const MouseButtonEventCallback& cb) {
    mbutton_cb = cb;
}

void GLFWWindow::set_scroll_event_callback(const ScrollEventCallback& cb) {
    scroll_cb = cb;
}

void GLFWWindow::set_key_event_callback(const KeyEventCallback& cb) {
    key_cb = cb;
}

GLFWWindow::operator GLFWwindow*() const {
    return window;
}

GLFWwindow* GLFWWindow::get_window() const {
    return window;
}

} // namespace merian
