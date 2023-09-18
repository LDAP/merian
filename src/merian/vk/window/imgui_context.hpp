#pragma once

#include "imgui.h"
#include "imgui_internal.h"

#include <memory>

namespace merian {

// A wrapper around ImGuiContext to simplify automatic cleanup.
class ImGuiContextWrapper : public std::enable_shared_from_this<ImGuiContextWrapper> {
  public:
    explicit ImGuiContextWrapper() {
        ctx = ImGui::CreateContext();
    }

    ~ImGuiContextWrapper() {
        ImGui::DestroyContext(ctx);
    }

    operator const ImGuiContext*() const {
        return ctx;
    }

    operator ImGuiContext*() const {
        return ctx;
    }

    ImGuiContext* get() const {
        return ctx;
    }

    // makes this context the global context
    void set_current_context() {
        ImGui::SetCurrentContext(ctx);
    }

    ImGuiIO& get_io() {
        return ctx->IO;
    }

  private:
    ImGuiContext* ctx;
};

using ImGuiContextWrapperHandle = std::shared_ptr<ImGuiContextWrapper>;

} // namespace merian
