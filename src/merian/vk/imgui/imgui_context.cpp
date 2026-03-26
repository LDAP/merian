#include "merian/vk/imgui/imgui_context.hpp"

#include "../../utils/fonts/jetbrains_mono.h"

namespace merian {

ImGuiContext::ImGuiContext() {
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
}

} // namespace merian
