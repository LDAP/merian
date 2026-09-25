#pragma once

#include "imgui.h"
#include "imgui_internal.h"
#include "merian/vk/imgui/imgui_theme.hpp"

#include <memory>

namespace merian {

// Points merian-core's copy of ImGui's globals at ctx. merian and merian-graph each link ImGui
// statically and so carry their own GImGui; every context switch has to reach both.
void imgui_set_core_context(::ImGuiContext* ctx);

// Owns the lifetime of a Dear ImGui context and provides a safe multi-context API.
//
// Use with_context() to run ImGui calls against this specific context.
class ImGuiContext : public std::enable_shared_from_this<ImGuiContext> {
  public:
    // Creates the Dear ImGui context and loads JetBrainsMono as the default font.
    ImGuiContext(ImGuiTheme theme = ImGuiTheme::Mocha);

    // Defined in merian-core: the settings handlers a context carries point into the binary that
    // created it, and Shutdown() calls them through that binary's GImGui.
    ~ImGuiContext();

    ::ImGuiContext* get() const {
        return ctx;
    }

    ImGuiIO& get_io() {
        return ctx->IO;
    }

    ImGuiStyle& get_style() {
        return ctx->Style;
    }

    // Temporarily sets this as the global Dear ImGui context, runs fn(), then restores previous.
    template <typename Fn> void with_context(Fn&& fn) {
        ::ImGuiContext* prev = ImGui::GetCurrentContext();
        set_current(ctx);
        fn();
        set_current(prev);
    }

    // Sets the context in this binary and in merian-core.
    static void set_current(::ImGuiContext* ctx) {
        ImGui::SetCurrentContext(ctx);
        imgui_set_core_context(ctx);
    }

    operator ::ImGuiContext*() const {
        return ctx;
    }

  private:
    ::ImGuiContext* ctx;
};

using ImGuiContextHandle = std::shared_ptr<ImGuiContext>;

} // namespace merian
