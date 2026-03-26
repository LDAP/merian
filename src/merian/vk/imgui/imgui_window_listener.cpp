#include "merian/vk/imgui/imgui_window_listener.hpp"
#include "merian/vk/imgui/imgui_context.hpp"

namespace merian {

ImGuiWindowListener::ImGuiWindowListener(ImGuiContextHandle ctx) : ctx(std::move(ctx)) {}

void ImGuiWindowListener::on_resize(vk::Extent2D framebuffer_extent, vk::Extent2D window_extent) {
    ImGuiIO& io = ctx->get_io();
    io.DisplaySize =
        ImVec2(static_cast<float>(window_extent.width), static_cast<float>(window_extent.height));
    if (window_extent.width > 0 && window_extent.height > 0) {
        io.DisplayFramebufferScale = ImVec2(static_cast<float>(framebuffer_extent.width) /
                                                static_cast<float>(window_extent.width),
                                            static_cast<float>(framebuffer_extent.height) /
                                                static_cast<float>(window_extent.height));
    }
}

void ImGuiWindowListener::on_display_scale_changed(const float display_scale) {
    ctx->get_io().DisplayFramebufferScale = ImVec2(display_scale, display_scale);
}

void ImGuiWindowListener::on_focus_changed(const bool focused) {
    ctx->get_io().AddFocusEvent(focused);
}

} // namespace merian
