#include "glfw_imgui.hpp"

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "imgui.h"

namespace merian {

GLFWImGui::GLFWImGui(const SharedContext context,
                     const bool no_mouse_cursor_change,
                     const vk::ImageLayout initial_layout)
    : context(context), no_mouse_cursor_change(no_mouse_cursor_change),
      initial_layout(initial_layout) {}

GLFWImGui::~GLFWImGui() {
    if (imgui_initialized) {
        context->device.waitIdle();

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        context->device.destroyDescriptorPool(imgui_pool);
    }

    for (auto& framebuffer : framebuffers) {
        context->device.destroyFramebuffer(framebuffer);
    }
    if (render_pass) {
        context->device.destroyRenderPass(render_pass);
    }
}

void GLFWImGui::recreate_render_pass(SwapchainAcquireResult& aquire_result) {
    if (render_pass) {
        context->device.destroyRenderPass(render_pass);
    }

    vk::AttachmentDescription attachment_desc{
        {},
        aquire_result.surface_format.format,
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
        {},
        vk::AccessFlagBits::eColorAttachmentWrite,
    };
    vk::RenderPassCreateInfo info = {
        {}, 1, &attachment_desc, 1, &subpass, 1, &dependency,
    };

    render_pass = context->device.createRenderPass(info);
}

void GLFWImGui::upload_imgui_fonts() {
    CommandPool pool(context->get_queue_GCT());

    auto cmd = pool.create_and_begin();
    ImGui_ImplVulkan_CreateFontsTexture(cmd);
    pool.end_all();
    context->get_queue_GCT()->submit_wait(cmd);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void GLFWImGui::init_imgui(GLFWwindow* window, SwapchainAcquireResult& aquire_result) {
    assert(render_pass);

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls
    if (no_mouse_cursor_change)
        io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;

    std::vector<vk::DescriptorPoolSize> pool_sizes = {
        {vk::DescriptorType::eSampler, 1000},
        {vk::DescriptorType::eCombinedImageSampler, 1000},
        {vk::DescriptorType::eSampledImage, 1000},
        {vk::DescriptorType::eStorageImage, 1000},
        {vk::DescriptorType::eUniformTexelBuffer, 1000},
        {vk::DescriptorType::eStorageTexelBuffer, 1000},
        {vk::DescriptorType::eUniformBuffer, 1000},
        {vk::DescriptorType::eStorageBuffer, 1000},
        {vk::DescriptorType::eUniformBufferDynamic, 1000},
        {vk::DescriptorType::eStorageBufferDynamic, 1000},
        {vk::DescriptorType::eInputAttachment, 1000}};

    vk::DescriptorPoolCreateInfo pool_info{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                           1000 * (uint32_t)pool_sizes.size(), pool_sizes};

    imgui_pool = context->device.createDescriptorPool(pool_info);

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = context->instance;
    init_info.PhysicalDevice = context->pd_container.physical_device;
    init_info.Device = context->device;
    init_info.QueueFamily = context->queue_family_idx_GCT;
    init_info.Queue = context->get_queue_GCT()->get_queue();
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = imgui_pool;
    init_info.Subpass = 0;
    init_info.MinImageCount = aquire_result.min_images;
    init_info.ImageCount = aquire_result.num_images;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.Allocator = VK_NULL_HANDLE;
    init_info.CheckVkResultFn = nullptr;

    ImGui_ImplVulkan_Init(&init_info, render_pass);

    upload_imgui_fonts();

    imgui_initialized = true;
}

// Start a new ImGui frame and renderpass. Returns the framebuffer.
vk::Framebuffer GLFWImGui::new_frame(vk::CommandBuffer& cmd,
                                     GLFWwindow* window,
                                     SwapchainAcquireResult& aquire_result) {
    if (aquire_result.did_recreate) {
        for (auto& framebuffer : framebuffers) {
            if (framebuffer)
                context->device.destroyFramebuffer(framebuffer);
        }
        framebuffers.assign(aquire_result.num_images, VK_NULL_HANDLE);
        recreate_render_pass(aquire_result);
    }

    if (!imgui_initialized) {
        init_imgui(window, aquire_result);
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (!framebuffers[aquire_result.index]) {
        vk::FramebufferCreateInfo fb_create_info = {
            {},
            render_pass,
            1,
            &aquire_result.view,
            aquire_result.extent.width,
            aquire_result.extent.height,
            1,
        };
        framebuffers[aquire_result.index] = context->device.createFramebuffer(fb_create_info);
    }
    vk::Framebuffer& framebuffer = framebuffers[aquire_result.index];

    vk::RenderPassBeginInfo rp_info = {
        render_pass, framebuffer, {{}, aquire_result.extent}, 0, nullptr};
    cmd.beginRenderPass(rp_info, vk::SubpassContents::eInline);

    return framebuffer;
}

// Render the ImGui to the current swapchain image
void GLFWImGui::render(vk::CommandBuffer& cmd) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    cmd.endRenderPass();
}

} // namespace merian
