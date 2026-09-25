#include "merian/vk/imgui/imgui_context.hpp"

#include "../../utils/fonts/jetbrains_mono.h"

namespace merian {

void imgui_set_core_context(::ImGuiContext* ctx) {
    ImGui::SetCurrentContext(ctx);
}

ImGuiContext::ImGuiContext(const ImGuiTheme theme) {
    // CreateContext leaves the new context current when there was none
    ::ImGuiContext* const prev = ImGui::GetCurrentContext();
    ctx = ImGui::CreateContext();

    // If we're not using freetype (which does not need oversampling) force higher oversampling for
    // crisper fonts.
    ImFontConfig font_config;
    font_config.OversampleH = 3;
    font_config.OversampleV = 1;
    font_config.FontDataOwnedByAtlas = false;
    ImFont* font = ctx->IO.Fonts->AddFontFromMemoryCompressedTTF(
        JetBrainsMono_compressed_data, JetBrainsMono_compressed_size, 16.0f, &font_config);
    ctx->IO.FontDefault = font;

    apply_imgui_theme(ctx->Style, theme);

    ImGui::SetCurrentContext(prev);
}

ImGuiContext::~ImGuiContext() {
    ImGui::DestroyContext(ctx);
}

} // namespace merian
