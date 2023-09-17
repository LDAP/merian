#pragma once

#include "merian/vk/window/swapchain.hpp"

namespace merian {

class GLFWImGui {

  public:
    // Set no_mouse_cursor_change to true if GLFWImGui is interfering with your cursor.
    // `initial_layout` which layout the swapchain image has when calling "new_frame".
    // initialize_context == true: constructor and destructor initialize and destroy the ImGui context.
    GLFWImGui(const SharedContext context,
              const bool no_mouse_cursor_change = false,
              const vk::ImageLayout initial_layout = vk::ImageLayout::ePresentSrcKHR,
              const bool initialize_context = true);
    ~GLFWImGui();

    // Start a new ImGui frame
    vk::Framebuffer
    new_frame(vk::CommandBuffer& cmd, GLFWwindow* window, SwapchainAcquireResult& aquire_result);

    // Render the ImGui to the current swapchain image
    void render(vk::CommandBuffer& cmd);

  private:
    void upload_imgui_fonts();
    void init_imgui(GLFWwindow* window, SwapchainAcquireResult& aquire_result);
    void recreate_render_pass(SwapchainAcquireResult& aquire_result);

  private:
    const SharedContext context;
    const bool no_mouse_cursor_change;
    const vk::ImageLayout initial_layout;
    const bool initialize_context;

    bool imgui_initialized = false;
    vk::DescriptorPool imgui_pool{VK_NULL_HANDLE};
    vk::RenderPass render_pass{VK_NULL_HANDLE};
    std::vector<vk::Framebuffer> framebuffers;
};

} // namespace merian
