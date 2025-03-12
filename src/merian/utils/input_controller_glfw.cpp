#include "merian/utils/input_controller_glfw.hpp"

namespace merian {

static std::unordered_map<GLFWwindow*, GLFWInputController*> window_controller_map;

int mods_from_glfw(int glfw_mods) {
    int mods = 0;
    mods |= glfw_mods & GLFW_MOD_SHIFT ? GLFWInputController::SHIFT : 0;
    mods |= glfw_mods & GLFW_MOD_CONTROL ? GLFWInputController::CONTROL : 0;
    mods |= glfw_mods & GLFW_MOD_ALT ? GLFWInputController::ALT : 0;
    mods |= glfw_mods & GLFW_MOD_SUPER ? GLFWInputController::SUPER : 0;
    mods |= glfw_mods & GLFW_MOD_CAPS_LOCK ? GLFWInputController::CAPS_LOCK : 0;
    mods |= glfw_mods & GLFW_MOD_NUM_LOCK ? GLFWInputController::NUM_LOCK : 0;
    return mods;
}

GLFWInputController::KeyStatus status_from_glfw(int action) {
    GLFWInputController::KeyStatus status;
    switch (action) {
    case GLFW_PRESS:
        status = GLFWInputController::PRESS;
        break;
    case GLFW_RELEASE:
        status = GLFWInputController::RELEASE;
        break;
    case GLFW_REPEAT:
        status = GLFWInputController::REPEAT;
        break;
    default:
        throw std::invalid_argument{"action not supported"};
    }
    return status;
}

void glfw_cursor_cb(GLFWwindow* window, double xpos, double ypos) {
    GLFWInputController* c = window_controller_map[window];

    if (!c->cursor_cb || !c->active)
        return;

    c->cursor_cb(*c, xpos, ypos);
}

void glfw_mouseb_cb(GLFWwindow* window, int glfw_button, int action, int glfw_mods) {
    GLFWInputController* c = window_controller_map[window];

    if (!c->mbutton_cb || !c->active)
        return;

    GLFWInputController::MouseButton button;
    switch (glfw_button) {
    case GLFW_MOUSE_BUTTON_1:
        button = GLFWInputController::MOUSE1;
        break;
    case GLFW_MOUSE_BUTTON_2:
        button = GLFWInputController::MOUSE2;
        break;
    case GLFW_MOUSE_BUTTON_3:
        button = GLFWInputController::MOUSE3;
        break;
    case GLFW_MOUSE_BUTTON_4:
        button = GLFWInputController::MOUSE4;
        break;
    case GLFW_MOUSE_BUTTON_5:
        button = GLFWInputController::MOUSE5;
        break;
    default:
        throw std::invalid_argument{"glfw_button not allowed"};
    }
    c->mbutton_cb(*c, button, status_from_glfw(action), mods_from_glfw(glfw_mods));
}

void glfw_key_cb(GLFWwindow* window, int key, int scancode, int action, int glfw_mods) {
    GLFWInputController* c = window_controller_map[window];

    if (!c->key_cb || !c->active)
        return;

    c->key_cb(*c, key, scancode, status_from_glfw(action), mods_from_glfw(glfw_mods));
}

void glfw_scroll_cb(GLFWwindow* window, double xoffset, double yoffset) {
    GLFWInputController* c = window_controller_map[window];

    if (!c->scroll_cb || !c->active)
        return;

    c->scroll_cb(*c, xoffset, yoffset);
}

GLFWInputController::GLFWInputController(const GLFWWindowHandle window) : window(window) {
    if (window_controller_map.contains(*window)) {
        throw std::runtime_error("there exists already an GLFWInputController for this window");
    }

    window_controller_map[*window] = this;

    // Setup callbacks to us
    glfwSetCursorPosCallback(*window, glfw_cursor_cb);
    glfwSetMouseButtonCallback(*window, glfw_mouseb_cb);
    glfwSetKeyCallback(*window, glfw_key_cb);
    glfwSetScrollCallback(*window, glfw_scroll_cb);
}

GLFWInputController::~GLFWInputController() {
    window_controller_map.erase(*window);
}

bool GLFWInputController::request_raw_mouse_input(bool enable) {
    if (glfwRawMouseMotionSupported() == 0)
        return false;

    if (enable) {
        glfwSetInputMode(*window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwSetInputMode(*window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    } else {
        glfwSetInputMode(*window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
        glfwSetInputMode(*window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
    return true;
}

// Returns true if raw mouse input is enabled.
bool GLFWInputController::get_raw_mouse_input() {
    return glfwGetInputMode(*window, GLFW_RAW_MOUSE_MOTION) != 0;
}

// Clear all callbacks
void GLFWInputController::reset() {
    cursor_cb = nullptr;
    mbutton_cb = nullptr;
    key_cb = nullptr;
    scroll_cb = nullptr;
}

void GLFWInputController::set_active(bool active) {
    this->active = active;
}

void GLFWInputController::set_mouse_cursor_callback(MouseCursorEventCallback cb) {
    cursor_cb = cb;
}
void GLFWInputController::set_mouse_button_callback(MouseButtonEventCallback cb) {
    mbutton_cb = cb;
}
void GLFWInputController::set_scroll_event_callback(ScrollEventCallback cb) {
    scroll_cb = cb;
}
void GLFWInputController::set_key_event_callback(KeyEventCallback cb) {
    key_cb = cb;
}

} // namespace merian
