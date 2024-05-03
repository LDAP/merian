#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/window/window.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace merian {

class GLFWWindow : public Window {

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

    vk::Extent2D framebuffer_extent() override {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        return vk::Extent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    }

  private:
    const SharedContext context;
    GLFWwindow* window;
};

using GLFWWindowHandle = std::shared_ptr<GLFWWindow>;

} // namespace merian
