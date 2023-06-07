#pragma once

#include "merian/utils/vector_utils.hpp"
#include "merian/vk/extension/extension.hpp"
#include "merian/vk/swapchain/swapchain.hpp"

#include <spdlog/logger.h>
#include <spdlog/spdlog.h>

namespace merian {

/*
 * @brief      Initializes GLFW and makes sure the graphics queue supports present.
 *
 */
class ExtensionVkGLFW : public Extension {

  public:
    ExtensionVkGLFW(int width = 1280, int height = 720, const char* title = "")
        : Extension("ExtensionVkGLFW") {
        if (!glfwInit())
            throw std::runtime_error("GLFW initialization failed!");
        if (!glfwVulkanSupported())
            throw std::runtime_error("GLFW reports to have no Vulkan support! Maybe it couldn't "
                                     "find the Vulkan loader!");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    }

    ~ExtensionVkGLFW() {}

    std::vector<const char*> required_instance_extension_names() const override {
        std::vector<const char*> required_extensions;
        uint32_t count;
        const char** extensions = glfwGetRequiredInstanceExtensions(&count);
        required_extensions.insert(required_extensions.end(), extensions, extensions + count);
        return required_extensions;
    }

    std::vector<const char*> required_device_extension_names(vk::PhysicalDevice) const override {
        return {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };
    }

    void on_instance_created(const vk::Instance&) override;
    bool accept_graphics_queue(const vk::PhysicalDevice&, std::size_t) override;
    void on_physical_device_selected(const Context::PhysicalDeviceContainer& pd_container) override;
    void on_context_created(const SharedContext context) override {
        weak_context = context;
    }
    void on_destroy_instance(const vk::Instance&) override;

    // Get the window and surface that was created by this extension.
    // This can be called EXACTLY ONCE!
    std::tuple<GLFWWindowHandle, SurfaceHandle> get() {
        if (!window) {
            std::runtime_error{"ExtensionVkGLFW:get() can only be called exactly once!"};
        }

        GLFWwindow* window = this->window;
        vk::SurfaceKHR surface = this->surface;

        this->window = nullptr;
        this->surface = vk::SurfaceKHR();

        assert(!weak_context.expired());
        SharedContext context = weak_context.lock();

        return {std::make_shared<GLFWWindow>(context, window),
                std::make_shared<Surface>(context, surface)};
    }

  private:
    vk::PhysicalDevice physical_device = VK_NULL_HANDLE;
    std::weak_ptr<Context> weak_context;

    GLFWwindow* window;
    vk::SurfaceKHR surface;
};

} // namespace merian
