#pragma once

// clang-format off
#include <vk/extension/extension.hpp>
#include <GLFW/glfw3.h>
// clang-format on

class ExtensionVkGLFW : public Extension {
  public:
    /**
     * @param[in]  preferred_surface_formats  The preferred surface formats in decreasing priority
     * @param[in]  fallback_format            The fallback format if non of the preferred formats is available
     */
    ExtensionVkGLFW(int width = 1280, int height = 720, const char* title = "",
                    std::vector<vk::SurfaceFormatKHR> preferred_surface_formats = {vk::Format::eR8G8B8A8Srgb,
                                                                                   vk::Format::eB8G8R8A8Srgb},
                    vk::PresentModeKHR preferred_vsync_off_mode = vk::PresentModeKHR::eMailbox)
        : preferred_surface_formats(preferred_surface_formats), preferred_vsync_off_mode(preferred_vsync_off_mode) {
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed!");
        if (!glfwVulkanSupported())
            throw std::runtime_error(
                "GLFW reports to have no Vulkan support! Maybe it couldn't find the Vulkan loader!");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    }
    ~ExtensionVkGLFW() {
        glfwDestroyWindow(window);
    }
    std::string name() const override {
        return "ExtensionVkGLFW";
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
    void on_instance_created(const vk::Instance&) override;
    void on_destroy_instance(const vk::Instance&) override;
    bool accept_graphics_queue(const vk::PhysicalDevice&, std::size_t) override;
    void on_physical_device_selected(const vk::PhysicalDevice&) override;
    void on_device_created(const vk::Device&) override;
    void on_destroy_device(const vk::Device&) override;

    // own methods
    void recreate_swapchain();
    void set_vsync(bool state) {
        if (state != vsync) {
            recreate_swapchain();
        }
    }

  private:
    /* Destroys swapchain and image views */
    void destroy_swapchain();
    /* Destroys image views only (for recreate) */
    void destroy_image_views();

  private:
    std::vector<vk::SurfaceFormatKHR> preferred_surface_formats;
    vk::PresentModeKHR preferred_vsync_off_mode;

  public:
    GLFWwindow* window;
    vk::SurfaceKHR surface;
    vk::SurfaceFormatKHR surface_format;
    vk::Extent2D extent2D;
    vk::SwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<vk::Image> swapchain_images;
    std::vector<vk::ImageView> swapchain_image_views;
    bool vsync = false;
    vk::Device device = VK_NULL_HANDLE;
    vk::PhysicalDevice physical_device = VK_NULL_HANDLE;
};
