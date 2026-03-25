#include "merian/vk/extension/glfw/glfw_window.hpp"
#include "merian/vk/extension/glfw/extension_glfw.hpp"
#include "merian/vk/extension/glfw/glfw_surface.hpp"

#include <spdlog/spdlog.h>

namespace merian {

namespace {

// clang-format off
InputController::Key key_from_glfw(const int key) {
    using K = InputController::Key;
    switch (key) {
    case GLFW_KEY_SPACE:          return K::SPACE;
    case GLFW_KEY_APOSTROPHE:     return K::APOSTROPHE;
    case GLFW_KEY_COMMA:          return K::COMMA;
    case GLFW_KEY_MINUS:          return K::MINUS;
    case GLFW_KEY_PERIOD:         return K::PERIOD;
    case GLFW_KEY_SLASH:          return K::SLASH;
    case GLFW_KEY_0:              return K::NUM_0;
    case GLFW_KEY_1:              return K::NUM_1;
    case GLFW_KEY_2:              return K::NUM_2;
    case GLFW_KEY_3:              return K::NUM_3;
    case GLFW_KEY_4:              return K::NUM_4;
    case GLFW_KEY_5:              return K::NUM_5;
    case GLFW_KEY_6:              return K::NUM_6;
    case GLFW_KEY_7:              return K::NUM_7;
    case GLFW_KEY_8:              return K::NUM_8;
    case GLFW_KEY_9:              return K::NUM_9;
    case GLFW_KEY_SEMICOLON:      return K::SEMICOLON;
    case GLFW_KEY_EQUAL:          return K::EQUAL;
    case GLFW_KEY_A:              return K::A;
    case GLFW_KEY_B:              return K::B;
    case GLFW_KEY_C:              return K::C;
    case GLFW_KEY_D:              return K::D;
    case GLFW_KEY_E:              return K::E;
    case GLFW_KEY_F:              return K::F;
    case GLFW_KEY_G:              return K::G;
    case GLFW_KEY_H:              return K::H;
    case GLFW_KEY_I:              return K::I;
    case GLFW_KEY_J:              return K::J;
    case GLFW_KEY_K:              return K::K;
    case GLFW_KEY_L:              return K::L;
    case GLFW_KEY_M:              return K::M;
    case GLFW_KEY_N:              return K::N;
    case GLFW_KEY_O:              return K::O;
    case GLFW_KEY_P:              return K::P;
    case GLFW_KEY_Q:              return K::Q;
    case GLFW_KEY_R:              return K::R;
    case GLFW_KEY_S:              return K::S;
    case GLFW_KEY_T:              return K::T;
    case GLFW_KEY_U:              return K::U;
    case GLFW_KEY_V:              return K::V;
    case GLFW_KEY_W:              return K::W;
    case GLFW_KEY_X:              return K::X;
    case GLFW_KEY_Y:              return K::Y;
    case GLFW_KEY_Z:              return K::Z;
    case GLFW_KEY_LEFT_BRACKET:   return K::LEFT_BRACKET;
    case GLFW_KEY_BACKSLASH:      return K::BACKSLASH;
    case GLFW_KEY_RIGHT_BRACKET:  return K::RIGHT_BRACKET;
    case GLFW_KEY_GRAVE_ACCENT:   return K::GRAVE_ACCENT;
    case GLFW_KEY_ESCAPE:         return K::ESCAPE;
    case GLFW_KEY_ENTER:          return K::ENTER;
    case GLFW_KEY_TAB:            return K::TAB;
    case GLFW_KEY_BACKSPACE:      return K::BACKSPACE;
    case GLFW_KEY_INSERT:         return K::INSERT;
    case GLFW_KEY_DELETE:         return K::DELETE;
    case GLFW_KEY_RIGHT:          return K::RIGHT;
    case GLFW_KEY_LEFT:           return K::LEFT;
    case GLFW_KEY_DOWN:           return K::DOWN;
    case GLFW_KEY_UP:             return K::UP;
    case GLFW_KEY_PAGE_UP:        return K::PAGE_UP;
    case GLFW_KEY_PAGE_DOWN:      return K::PAGE_DOWN;
    case GLFW_KEY_HOME:           return K::HOME;
    case GLFW_KEY_END:            return K::END;
    case GLFW_KEY_CAPS_LOCK:      return K::CAPS_LOCK;
    case GLFW_KEY_SCROLL_LOCK:    return K::SCROLL_LOCK;
    case GLFW_KEY_NUM_LOCK:       return K::NUM_LOCK;
    case GLFW_KEY_PRINT_SCREEN:   return K::PRINT_SCREEN;
    case GLFW_KEY_PAUSE:          return K::PAUSE;
    case GLFW_KEY_F1:             return K::F1;
    case GLFW_KEY_F2:             return K::F2;
    case GLFW_KEY_F3:             return K::F3;
    case GLFW_KEY_F4:             return K::F4;
    case GLFW_KEY_F5:             return K::F5;
    case GLFW_KEY_F6:             return K::F6;
    case GLFW_KEY_F7:             return K::F7;
    case GLFW_KEY_F8:             return K::F8;
    case GLFW_KEY_F9:             return K::F9;
    case GLFW_KEY_F10:            return K::F10;
    case GLFW_KEY_F11:            return K::F11;
    case GLFW_KEY_F12:            return K::F12;
    case GLFW_KEY_KP_0:           return K::KP_0;
    case GLFW_KEY_KP_1:           return K::KP_1;
    case GLFW_KEY_KP_2:           return K::KP_2;
    case GLFW_KEY_KP_3:           return K::KP_3;
    case GLFW_KEY_KP_4:           return K::KP_4;
    case GLFW_KEY_KP_5:           return K::KP_5;
    case GLFW_KEY_KP_6:           return K::KP_6;
    case GLFW_KEY_KP_7:           return K::KP_7;
    case GLFW_KEY_KP_8:           return K::KP_8;
    case GLFW_KEY_KP_9:           return K::KP_9;
    case GLFW_KEY_KP_DECIMAL:     return K::KP_DECIMAL;
    case GLFW_KEY_KP_DIVIDE:      return K::KP_DIVIDE;
    case GLFW_KEY_KP_MULTIPLY:    return K::KP_MULTIPLY;
    case GLFW_KEY_KP_SUBTRACT:    return K::KP_SUBTRACT;
    case GLFW_KEY_KP_ADD:         return K::KP_ADD;
    case GLFW_KEY_KP_ENTER:       return K::KP_ENTER;
    case GLFW_KEY_KP_EQUAL:       return K::KP_EQUAL;
    case GLFW_KEY_LEFT_SHIFT:     return K::LEFT_SHIFT;
    case GLFW_KEY_LEFT_CONTROL:   return K::LEFT_CTRL;
    case GLFW_KEY_LEFT_ALT:       return K::LEFT_ALT;
    case GLFW_KEY_LEFT_SUPER:     return K::LEFT_SUPER;
    case GLFW_KEY_RIGHT_SHIFT:    return K::RIGHT_SHIFT;
    case GLFW_KEY_RIGHT_CONTROL:  return K::RIGHT_CTRL;
    case GLFW_KEY_RIGHT_ALT:      return K::RIGHT_ALT;
    case GLFW_KEY_RIGHT_SUPER:    return K::RIGHT_SUPER;
    case GLFW_KEY_MENU:           return K::MENU;
    default:                      return K::UNKNOWN;
    }
}
// clang-format on

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

// clang-format off
InputController::KeyStatus status_from_glfw(const int action) {
    switch (action) {
    case GLFW_PRESS:   return InputController::KeyStatus::PRESS;
    case GLFW_RELEASE: return InputController::KeyStatus::RELEASE;
    case GLFW_REPEAT:  return InputController::KeyStatus::REPEAT;
    default:           return InputController::KeyStatus::UNKNOWN;
    }
}

InputController::MouseButton mouse_button_from_glfw(const int glfw_button) {
    switch (glfw_button) {
    case GLFW_MOUSE_BUTTON_1: return InputController::MouseButton::MOUSE1;
    case GLFW_MOUSE_BUTTON_2: return InputController::MouseButton::MOUSE2;
    case GLFW_MOUSE_BUTTON_3: return InputController::MouseButton::MOUSE3;
    case GLFW_MOUSE_BUTTON_4: return InputController::MouseButton::MOUSE4;
    case GLFW_MOUSE_BUTTON_5: return InputController::MouseButton::MOUSE5;
    default:                  return InputController::MouseButton::UNKNOWN;
    }
}
// clang-format on

} // namespace

void GLFWWindow::cb_close(GLFWwindow* w) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    if (self->close_cb)
        self->close_cb();
}

void GLFWWindow::cb_cursor(GLFWwindow* w, const double xpos, const double ypos) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    self->dispatch_cursor(xpos, ypos);
}

void GLFWWindow::cb_mouse_button(GLFWwindow* w,
                                 const int glfw_button,
                                 const int action,
                                 const int glfw_mods) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    // dispatch mods
    self->dispatch_key(Key::UNKNOWN, KeyStatus::UNKNOWN, mods_from_glfw(glfw_mods));
    self->dispatch_mouse_button(mouse_button_from_glfw(glfw_button), status_from_glfw(action));
}

void GLFWWindow::cb_key(
    GLFWwindow* w, const int key, const int /*scancode*/, const int action, const int glfw_mods) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    self->dispatch_key(key_from_glfw(key), status_from_glfw(action), mods_from_glfw(glfw_mods));
}

void GLFWWindow::cb_scroll(GLFWwindow* w, const double xoffset, const double yoffset) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    self->dispatch_scroll(xoffset, yoffset);
}

void GLFWWindow::cb_char(GLFWwindow* w, const unsigned int codepoint) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    self->dispatch_char(codepoint);
}

void GLFWWindow::cb_framebuffer_size(GLFWwindow* w, const int width, const int height) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    int ww, wh;
    glfwGetWindowSize(w, &ww, &wh);
    self->dispatch_resize({static_cast<uint32_t>(width), static_cast<uint32_t>(height)},
                          {static_cast<uint32_t>(ww), static_cast<uint32_t>(wh)});
}

void GLFWWindow::cb_content_scale(GLFWwindow* w,
                                  const float xscale,
                                  [[maybe_unused]] const float yscale) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    self->dispatch_display_scale_changed(xscale);
}

void GLFWWindow::cb_focus(GLFWwindow* w, const int focused) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    self->dispatch_focus_changed(focused != 0);
}

void GLFWWindow::cb_iconify(GLFWwindow* w, const int iconified) {
    auto* self = static_cast<GLFWWindow*>(glfwGetWindowUserPointer(w));
    if (iconified != 0)
        self->dispatch_minimized();
    else
        self->dispatch_restored();
}

GLFWWindow::GLFWWindow(const DeviceHandle& device, const WindowCreateInfo& create_info)
    : device(device) {
    SPDLOG_DEBUG("create window ({})", fmt::ptr(this));
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_SCALE_TO_MONITOR, GLFW_TRUE);
    window = glfwCreateWindow(create_info.width, create_info.height, create_info.title.c_str(),
                              nullptr, nullptr);

    glfwSetWindowUserPointer(window, this);
    glfwSetWindowCloseCallback(window, cb_close);
    glfwSetCursorPosCallback(window, cb_cursor);
    glfwSetMouseButtonCallback(window, cb_mouse_button);
    glfwSetKeyCallback(window, cb_key);
    glfwSetScrollCallback(window, cb_scroll);
    glfwSetCharCallback(window, cb_char);
    glfwSetFramebufferSizeCallback(window, cb_framebuffer_size);
    glfwSetWindowContentScaleCallback(window, cb_content_scale);
    glfwSetWindowFocusCallback(window, cb_focus);
    glfwSetWindowIconifyCallback(window, cb_iconify);

    // clang-format off
    cursors[static_cast<size_t>(WindowCursorShape::Arrow)]      = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
    cursors[static_cast<size_t>(WindowCursorShape::TextInput)]  = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeAll)]  = glfwCreateStandardCursor(GLFW_RESIZE_ALL_CURSOR);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeNS)]   = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeEW)]   = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeNESW)] = glfwCreateStandardCursor(GLFW_RESIZE_NESW_CURSOR);
    cursors[static_cast<size_t>(WindowCursorShape::ResizeNWSE)] = glfwCreateStandardCursor(GLFW_RESIZE_NWSE_CURSOR);
    cursors[static_cast<size_t>(WindowCursorShape::Hand)]       = glfwCreateStandardCursor(GLFW_POINTING_HAND_CURSOR);
    cursors[static_cast<size_t>(WindowCursorShape::NotAllowed)] = glfwCreateStandardCursor(GLFW_NOT_ALLOWED_CURSOR);
    // clang-format on
}

GLFWWindow::~GLFWWindow() {
    SPDLOG_DEBUG("destroy window ({})", fmt::ptr(this));
    for (GLFWcursor* c : cursors)
        if (c)
            glfwDestroyCursor(c);
    glfwDestroyWindow(window);

    // Flush the destroy to the compositor so the window disappears immediately,
    glfwPollEvents();
}

vk::Extent2D GLFWWindow::framebuffer_extent() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    return vk::Extent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
}

vk::Extent2D GLFWWindow::window_extent() {
    int width, height;
    glfwGetWindowSize(window, &width, &height);
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
    try {
        glfwPollEvents();
    } catch (merian::ExtensionGLFW::glfw_error& e) {
        SPDLOG_ERROR("glfw poll events failed: {}", e.what());
    }
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

void GLFWWindow::set_cursor_pos(const double x, const double y) {
    glfwSetCursorPos(window, x, y);
}

void GLFWWindow::set_cursor(const WindowCursorShape shape) {
    if (shape == current_cursor_shape)
        return;
    current_cursor_shape = shape;

    if (is_mouse_grabbed())
        return;

    if (shape == WindowCursorShape::Hidden) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    } else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        glfwSetCursor(window, cursors[static_cast<size_t>(shape)]);
    }
}

bool GLFWWindow::set_mouse_grabbed(const bool grabbed) {
    if (grabbed) {
        glfwSetCursor(window, nullptr); // clear stale cursor before disabling (Wayland workaround)
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        if (glfwRawMouseMotionSupported() == GLFW_TRUE)
            glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    } else {
        if (glfwRawMouseMotionSupported() == GLFW_TRUE)
            glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
        if (current_cursor_shape == WindowCursorShape::Hidden) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            glfwSetCursor(window, cursors[static_cast<size_t>(current_cursor_shape)]);
        }
    }
    return true;
}

bool GLFWWindow::is_mouse_grabbed() const {
    return glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
}

float GLFWWindow::get_display_scale() {
    float xscale = 1.0f;
    glfwGetWindowContentScale(window, &xscale, nullptr);
    return xscale;
}

void GLFWWindow::set_clipboard_text(const char* text) {
    // there is a single global lock on windows which sometimes cannot be acqired.
    const uint32_t tries = 5;
    uint32_t t = 0;
    while (true) {
        try {
            glfwSetClipboardString(nullptr, text);
            return;
        } catch (merian::ExtensionGLFW::glfw_error& e) {
            if (t++ > tries) {
                SPDLOG_ERROR("glfw failed to set the clipboard: {}", e.what());
                return;
            }
        }
    }
}

const char* GLFWWindow::get_clipboard_text() {
    // there is a single global lock on windows which sometimes cannot be acqired.
    const uint32_t tries = 5;
    uint32_t t = 0;
    while (true) {
        try {
            return glfwGetClipboardString(nullptr);
        } catch (merian::ExtensionGLFW::glfw_error& e) {
            if (t++ > tries) {
                SPDLOG_ERROR("glfw failed to set the clipboard: {}", e.what());
                return nullptr;
            }
        }
    }
}

GLFWWindow::operator GLFWwindow*() const {
    return window;
}

GLFWwindow* GLFWWindow::get_window() const {
    return window;
}

} // namespace merian
