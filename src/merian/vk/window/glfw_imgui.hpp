#pragma once

#include "merian/vk/window/imgui_context.hpp"
#include "merian/vk/window/swapchain.hpp"
#include "imgui.h"

namespace merian {

static const std::vector<vk::DescriptorPoolSize> MERIAN_GLFW_IMGUI_DEFAULT_POOL_SIZES = {
    // enough to fit a few fonts
    {vk::DescriptorType::eCombinedImageSampler, 8},
};

class GLFWImGui {
  public:
    // Set no_mouse_cursor_change to true if GLFWImGui is interfering with your cursor.
    // `initial_layout` which layout the swapchain image has when calling "new_frame".
    // initialize_context == true: constructor and destructor initialize and destroy the ImGui
    // context.
    // Adapt pool sizes to your needs (eg to fit all fonts).
    // Make sure to add all fonts before calling new_frame
    GLFWImGui(const SharedContext& context,
              const ImGuiContextWrapperHandle& ctx,
              const bool no_mouse_cursor_change = false,
              const vk::ImageLayout initial_layout = vk::ImageLayout::ePresentSrcKHR,
              const std::vector<vk::DescriptorPoolSize> pool_sizes =
                  MERIAN_GLFW_IMGUI_DEFAULT_POOL_SIZES);
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
    const ImGuiContextWrapperHandle ctx;

    const bool no_mouse_cursor_change;
    const vk::ImageLayout initial_layout;
    const std::vector<vk::DescriptorPoolSize> pool_sizes;

    bool imgui_initialized = false;
    vk::DescriptorPool imgui_pool{VK_NULL_HANDLE};
    vk::RenderPass render_pass{VK_NULL_HANDLE};
    std::vector<vk::Framebuffer> framebuffers;
};

} // namespace merian
