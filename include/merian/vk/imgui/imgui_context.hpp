#pragma once

#include "imgui.h"
#include "imgui_internal.h"

#include <memory>

namespace merian {

// Owns the lifetime of a Dear ImGui context and provides a safe multi-context API.
//
// Use with_context() to run Dear ImGui calls against this specific context.
// register_input_listener() / register_window_listener() are handled by ImGuiMerianBackend.
class ImGuiContext : public std::enable_shared_from_this<ImGuiContext> {
  public:
    // Creates the Dear ImGui context and loads JetBrainsMono as the default font.
    ImGuiContext();

    ~ImGuiContext() {
        ImGui::DestroyContext(ctx);
    }

    ::ImGuiContext* get() const {
        return ctx;
    }

    ImGuiIO& get_io() {
        return ctx->IO;
    }

    // Temporarily sets this as the global Dear ImGui context, runs fn(), then restores previous.
    template <typename Fn>
    void with_context(Fn&& fn) {
        ::ImGuiContext* prev = ImGui::GetCurrentContext();
        ImGui::SetCurrentContext(ctx);
        fn();
        ImGui::SetCurrentContext(prev);
    }

    operator ::ImGuiContext*() const {
        return ctx;
    }

  private:
    ::ImGuiContext* ctx;
};

using ImGuiContextHandle = std::shared_ptr<ImGuiContext>;

} // namespace merian
