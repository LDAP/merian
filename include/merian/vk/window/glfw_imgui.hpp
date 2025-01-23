#pragma once

#include "merian/vk/renderpass/framebuffer.hpp"

#include "merian/vk/window/imgui_context.hpp"
#include "merian/vk/window/swapchain_manager.hpp"
#include <GLFW/glfw3.h>

namespace merian {

static const std::vector<vk::DescriptorPoolSize> MERIAN_GLFW_IMGUI_DEFAULT_POOL_SIZES = {
    // enough to fit a few fonts
    {vk::DescriptorType::eCombinedImageSampler, 8},
};

// GLFW-Vulkan backend for ImGui.
class GLFWImGui {
  public:
    // Set no_mouse_cursor_change to true if GLFWImGui is interfering with your cursor.
    // `initial_layout` which layout the swapchain image has when calling "new_frame".
    // initialize_context == true: constructor and destructor initialize and destroy the ImGui
    // context.
    // Adapt pool sizes to your needs (eg to fit all fonts).
    // Make sure to add all fonts before calling new_frame
    GLFWImGui(const ContextHandle& context,
              const ImGuiContextWrapperHandle& ctx,
              const bool no_mouse_cursor_change = false,
              const vk::ImageLayout initial_layout = vk::ImageLayout::ePresentSrcKHR,
              const std::vector<vk::DescriptorPoolSize>& pool_sizes =
                  MERIAN_GLFW_IMGUI_DEFAULT_POOL_SIZES);
    ~GLFWImGui();

    // Start a new ImGui frame
    FramebufferHandle new_frame(QueueHandle& queue,
                                const CommandBufferHandle& cmd,
                                GLFWwindow* window,
                                const SwapchainAcquireResult& aquire_result);

    // Render the ImGui to the current swapchain image
    void render(const CommandBufferHandle& cmd);

  private:
    void init_imgui(GLFWwindow* window,
                    const SwapchainAcquireResult& aquire_result,
                    const QueueHandle& queue);

    void init_vulkan(const SwapchainAcquireResult& aquire_result, const QueueHandle& queue);

    void create_render_pass(const SwapchainAcquireResult& aquire_result);

  private:
    const ContextHandle context;
    const ImGuiContextWrapperHandle ctx;

    const bool no_mouse_cursor_change;
    const vk::ImageLayout initial_layout;
    const std::vector<vk::DescriptorPoolSize> pool_sizes;

    bool imgui_initialized = false;
    GLFWwindow* window; // only valid if initialized
    vk::DescriptorPool imgui_pool{VK_NULL_HANDLE};
    RenderPassHandle renderpass = nullptr;
    std::vector<FramebufferHandle> framebuffers;
    vk::SurfaceFormatKHR current_surface_format;
};

} // namespace merian
