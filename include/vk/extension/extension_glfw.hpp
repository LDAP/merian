#pragma once

// clang-format off
#include <vk/extension/extension.hpp>
#include <GLFW/glfw3.h>
// clang-format on

class ExtensionGLFW : public Extension {
  public:
    ExtensionGLFW(int width = 1280, int height = 720, const char* title = PROJECT_NAME) {
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed!");
        if (!glfwVulkanSupported())
            throw std::runtime_error(
                "GLFW reports to have no Vulkan support! Maybe it couldn't find the Vulkan loader!");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    }
    ~ExtensionGLFW() {
        glfwDestroyWindow(window);
    }
    std::string name() const override {
        return "ExtensionGLFW";
    }
    std::vector<const char*> required_instance_extension_names() const override {
        std::vector<const char*> required_extensions;
        uint32_t count;
        const char** extensions = glfwGetRequiredInstanceExtensions(&count);
        required_extensions.insert(required_extensions.end(), extensions, extensions + count);
        return required_extensions;
    }
    std::vector<const char*> required_device_extension_names() const override {
        return {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };
    }
    void on_instance_created(vk::Instance&) override;
    void on_destroy_instance(vk::Instance&) override;
    bool accept_graphics_queue(vk::PhysicalDevice&, std::size_t) override;
    void on_context_created(Context&) override;
    void on_destroy_context(Context& context) override;

    // own methods
    void recreate_swapchain(Context& context);
    /* Destroys swapchain and image views */
    void destroy_swapchain(Context& context);
    /* Destroys image views only (for recreate) */
    void destroy_image_views(Context& context);

  public:
    GLFWwindow* window;
    vk::SurfaceKHR surface;
    vk::SurfaceFormatKHR surface_format;
    vk::Extent2D extent2D;
    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<vk::Image> swapchain_images;
    std::vector<vk::ImageView> swapchain_image_views;
};
