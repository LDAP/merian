#pragma once

#include "merian/vk/context.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace merian {

class GLFWWindow : public std::enable_shared_from_this<GLFWWindow> {

  public:

    // Manage the supplied window. The window is destroyed when this object is destroyed.
    GLFWWindow(const SharedContext& context, GLFWwindow* window)
        : context(context), window(window) {}

    GLFWWindow(const SharedContext& context,
               int width = 1280,
               int height = 720,
               const char* title = "")
        : context(context) {
        // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        SPDLOG_DEBUG("create window ({})", fmt::ptr(this));
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    }

    ~GLFWWindow() {
        SPDLOG_DEBUG("destroy window ({})", fmt::ptr(this));
        glfwDestroyWindow(window);
    }

    operator GLFWwindow*() const {
        return window;
    }

    GLFWwindow* get_window() const {
        return window;
    }

  private:
    const SharedContext context;
    GLFWwindow* window;
};

using GLFWWindowHandle = std::shared_ptr<GLFWWindow>;

} // namespace merian
