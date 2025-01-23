#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/window/window.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace merian {

class GLFWWindow : public Window {
  public:
    friend class ExtensionGLFW;

  private:
    GLFWWindow(const ContextHandle& context,
               int width = 1280,
               int height = 720,
               const char* title = "")
        : context(context) {
        SPDLOG_DEBUG("create window ({})", fmt::ptr(this));
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    }

  public:
    ~GLFWWindow() {
        SPDLOG_DEBUG("destroy window ({})", fmt::ptr(this));
        glfwDestroyWindow(window);
    }

    bool should_close() const;

    operator GLFWwindow*() const;

    GLFWwindow* get_window() const;

    SurfaceHandle get_surface() override;

    vk::Extent2D framebuffer_extent() override;

  private:
    ContextHandle context;
    GLFWwindow* window = NULL;
};

using GLFWWindowHandle = std::shared_ptr<GLFWWindow>;

} // namespace merian
