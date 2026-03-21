#include "merian/vk/imgui/imgui_merian_backend.hpp"

#include "imgui.h"
#include "imgui_internal.h"

namespace merian {

namespace {

// clang-format off
WindowCursorShape cursor_shape_from_imgui_impl(const int imgui_cursor) {
    switch (imgui_cursor) {
    case ImGuiMouseCursor_Arrow:      return WindowCursorShape::Arrow;
    case ImGuiMouseCursor_TextInput:  return WindowCursorShape::TextInput;
    case ImGuiMouseCursor_ResizeAll:  return WindowCursorShape::ResizeAll;
    case ImGuiMouseCursor_ResizeNS:   return WindowCursorShape::ResizeNS;
    case ImGuiMouseCursor_ResizeEW:   return WindowCursorShape::ResizeEW;
    case ImGuiMouseCursor_ResizeNESW: return WindowCursorShape::ResizeNESW;
    case ImGuiMouseCursor_ResizeNWSE: return WindowCursorShape::ResizeNWSE;
    case ImGuiMouseCursor_Hand:       return WindowCursorShape::Hand;
    case ImGuiMouseCursor_NotAllowed: return WindowCursorShape::NotAllowed;
    default:                          return WindowCursorShape::Hidden;
    }
}
// clang-format on

} // namespace

ImGuiMerianBackend::ImGuiMerianBackend(const ImGuiContextHandle& ctx) : ctx(ctx) {
    ctx->get()->IO.BackendPlatformUserData = this;
    ctx->get()->IO.BackendPlatformName = "merian";
}

ImGuiMerianBackend::~ImGuiMerianBackend() {
    ::ImGuiContext* raw = ctx->get();
    if (raw->IO.BackendPlatformUserData == this) {
        raw->IO.BackendPlatformUserData = nullptr;
        raw->IO.BackendPlatformName = nullptr;
    }
}

void ImGuiMerianBackend::new_frame(const float delta_time) {
    ctx->get_io().DeltaTime = delta_time > 0.0f ? delta_time : (1.0f / 60.0f);
    ctx->with_context([] { ImGui::NewFrame(); });
}

// static
WindowCursorShape ImGuiMerianBackend::cursor_shape_from_imgui(const int imgui_cursor) {
    return cursor_shape_from_imgui_impl(imgui_cursor);
}

} // namespace merian
