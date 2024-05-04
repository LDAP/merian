#pragma once

#include "merian/vk/context.hpp"
#include "merian/vk/window/window.hpp"
#include "merian/vk/extension/extension_vk_glfw.hpp"

#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace merian {

class GLFWWindow : public Window {
  public:
    GLFWWindow(const SharedContext& context,
               int width = 1280,
               int height = 720,
               const char* title = "")
        : context(context) {
        if (!context->get_extension<ExtensionVkGLFW>()) {
            SPDLOG_WARN("ExtensionVkGLFW not found. You have to init GLFW manually!");
        }

        SPDLOG_DEBUG("create window ({})", fmt::ptr(this));
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    }

    ~GLFWWindow() {
        SPDLOG_DEBUG("destroy window ({})", fmt::ptr(this));
        glfwDestroyWindow(window);
    }

    operator GLFWwindow*() const;

    GLFWwindow* get_window() const;

    SurfaceHandle get_surface() override;

    vk::Extent2D framebuffer_extent() override;

  private:
    SharedContext context;
    GLFWwindow* window;
};

using GLFWWindowHandle = std::shared_ptr<GLFWWindow>;

} // namespace merian
