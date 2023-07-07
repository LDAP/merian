#pragma once

#include "merian/vk/window/swapchain.hpp"

namespace merian {

class GLFWImGui {

  public:
    GLFWImGui(const SharedContext context, const bool no_mouse_cursor_change = false);
    ~GLFWImGui();

    // Start a new ImGui frame
    vk::Framebuffer new_frame(vk::CommandBuffer& cmd, GLFWwindow* window, SwapchainAcquireResult& aquire_result);

    // Render the ImGui to the current swapchain image
    void render(vk::CommandBuffer& cmd);

private:
    void upload_imgui_fonts();
    void init_imgui(GLFWwindow* window, SwapchainAcquireResult& aquire_result);
    void recreate_render_pass(SwapchainAcquireResult& aquire_result);

  private:
    const SharedContext context;
    const bool no_mouse_cursor_change;

    bool imgui_initialized;
    vk::DescriptorPool imgui_pool;
    vk::RenderPass render_pass;
    std::vector<vk::Framebuffer> framebuffers;
};

} // namespace merian
