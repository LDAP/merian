#include "merian/vk/window/glfw_imgui.hpp"

#include "merian/vk/command/command_buffer.hpp"

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"

namespace merian {

GLFWImGui::GLFWImGui(const ContextHandle& context,
                     const ImGuiContextWrapperHandle& ctx,
                     const bool no_mouse_cursor_change,
                     const vk::ImageLayout initial_layout,
                     const std::vector<vk::DescriptorPoolSize>& pool_sizes)
    : context(context), ctx(ctx), no_mouse_cursor_change(no_mouse_cursor_change),
      initial_layout(initial_layout), pool_sizes(pool_sizes) {}

GLFWImGui::~GLFWImGui() {
    ImGuiContext* current_context = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(ctx->get());

    if (imgui_initialized) {
        context->device.waitIdle();

        ImGui_ImplVulkan_DestroyFontsTexture();
        ImGui_ImplVulkan_Shutdown();

        ImGui_ImplGlfw_RestoreCallbacks(window);
        ImGui_ImplGlfw_Shutdown();

        context->device.destroyDescriptorPool(imgui_pool);
    }

    ImGui::SetCurrentContext(current_context);
}

void GLFWImGui::create_render_pass(const SwapchainAcquireResult& aquire_result) {
    vk::AttachmentDescription attachment_desc{
        {},
        aquire_result.image_view->get_image()->get_format(),
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eLoad,
        vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare,
        initial_layout,
        vk::ImageLayout::ePresentSrcKHR,
    };
    vk::AttachmentReference color_attachment{
        0,
        vk::ImageLayout::eColorAttachmentOptimal,
    };
    vk::SubpassDescription subpass = {
        {}, vk::PipelineBindPoint::eGraphics, 0, nullptr, 1, &color_attachment,
    };
    vk::SubpassDependency dependency = {
        VK_SUBPASS_EXTERNAL,
        0,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::AccessFlagBits::eColorAttachmentWrite,
        vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite,
    };
    vk::RenderPassCreateInfo info = {
        {}, 1, &attachment_desc, 1, &subpass, 1, &dependency,
    };

    renderpass = RenderPass::create(context, info);
}

void GLFWImGui::init_vulkan(const SwapchainAcquireResult& aquire_result, const QueueHandle& queue) {
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context->instance;
    init_info.PhysicalDevice = context->physical_device.physical_device;
    init_info.Device = context->device;
    init_info.QueueFamily = queue->get_queue_family_index();
    init_info.Queue = queue->get_queue();
    init_info.PipelineCache = context->pipeline_cache;
    init_info.DescriptorPool = imgui_pool;
    init_info.Subpass = 0;
    init_info.MinImageCount = aquire_result.min_images;
    init_info.ImageCount = aquire_result.num_images;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = VK_NULL_HANDLE;
    init_info.CheckVkResultFn = nullptr;
    init_info.RenderPass = **renderpass;

    current_surface_format = aquire_result.image_view->get_image()->get_format();

    ImGui_ImplVulkan_Init(&init_info);

    ImGui_ImplVulkan_CreateFontsTexture();
}

void GLFWImGui::init_imgui(GLFWwindow* window,
                           const SwapchainAcquireResult& aquire_result,
                           const QueueHandle& queue) {
    assert(renderpass);

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
    if (no_mouse_cursor_change)
        io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;

    vk::DescriptorPoolCreateInfo pool_info{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1,
                                           pool_sizes};

    imgui_pool = context->device.createDescriptorPool(pool_info);

    ImGui_ImplGlfw_InitForVulkan(window, true);
    init_vulkan(aquire_result, queue);

    this->window = window;
    imgui_initialized = true;
}

// Start a new ImGui frame and renderpass. Returns the framebuffer.
FramebufferHandle GLFWImGui::new_frame(QueueHandle& queue,
                                       const CommandBufferHandle& cmd,
                                       GLFWwindow* window,
                                       const SwapchainAcquireResult& aquire_result) {
    ImGuiContext* current_context = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(ctx->get());

    if (aquire_result.did_recreate) {
        framebuffers.assign(aquire_result.num_images, VK_NULL_HANDLE);
        create_render_pass(aquire_result);
    }

    if (!imgui_initialized) {
        init_imgui(window, aquire_result, queue);
    } else if (aquire_result.did_recreate &&
               aquire_result.image_view->get_image()->get_format() != current_surface_format) {
        // Workaround: needs vulkan backend restart
        context->device.waitIdle();
        ImGui_ImplVulkan_DestroyFontsTexture();
        ImGui_ImplVulkan_Shutdown();
        init_vulkan(aquire_result, queue);
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (!framebuffers[aquire_result.index]) {
        framebuffers[aquire_result.index] = Framebuffer::create(
            context, renderpass, aquire_result.image_view->get_image()->get_extent(),
            aquire_result.image_view);
    }

    FramebufferHandle& framebuffer = framebuffers[aquire_result.index];

    cmd->begin_render_pass(framebuffer);

    ImGui::SetCurrentContext(current_context);
    return framebuffer;
}

// Render the ImGui to the current swapchain image
void GLFWImGui::render(const CommandBufferHandle& cmd) {
    ImGuiContext* current_context = ImGui::GetCurrentContext();
    ImGui::SetCurrentContext(ctx->get());

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd->get_command_buffer());
    cmd->end_render_pass();

    ImGui::SetCurrentContext(current_context);
}

} // namespace merian
