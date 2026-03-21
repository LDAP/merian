#include "merian/vk/imgui/imgui_context.hpp"

#include "../../utils/fonts/jetbrains_mono.h"

namespace merian {

ImGuiContext::ImGuiContext() {
    ctx = ImGui::CreateContext();
    ctx->IO.Fonts->AddFontFromMemoryCompressedTTF(JetBrainsMono_compressed_data,
                                                  JetBrainsMono_compressed_size, 16.0f);
}

} // namespace merian
