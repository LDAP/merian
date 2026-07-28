#include "merian/vk/imgui/imgui_window_listener.hpp"
#include "merian/vk/imgui/imgui_context.hpp"

namespace merian {

ImGuiWindowListener::ImGuiWindowListener(ImGuiContextHandle ctx) : ctx(std::move(ctx)) {}

void ImGuiWindowListener::on_focus_changed(const bool focused) {
    ctx->get_io().AddFocusEvent(focused);
}

} // namespace merian
