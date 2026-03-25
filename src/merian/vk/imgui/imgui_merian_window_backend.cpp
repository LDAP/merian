#include "merian/vk/imgui/imgui_merian_window_backend.hpp"
#include "merian/vk/imgui/imgui_input_listener.hpp"
#include "merian/vk/imgui/imgui_window_listener.hpp"

#include "imgui.h"
#include "imgui_internal.h"
#include <spdlog/spdlog.h>

namespace merian {

ImGuiMerianWindowBackend::ImGuiMerianWindowBackend(const ImGuiContextHandle& ctx,
                                                   const WindowHandle& win,
                                                   const int input_priority)
    : ImGuiMerianBackend(ctx), window(win) {
    input_listener = std::make_shared<ImGuiInputListener>(ctx);
    window_listener = std::make_shared<ImGuiWindowListener>(ctx);

    win->get_input_controller()->add_listener(input_listener, input_priority);
    win->add_window_listener(window_listener);

    ImGuiIO& io = ctx->get_io();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;

    ctx->get()->PlatformIO.Platform_SetClipboardTextFn = [](::ImGuiContext* c, const char* text) {
        auto* backend = static_cast<ImGuiMerianWindowBackend*>(c->IO.BackendPlatformUserData);
        if (backend && backend->window)
            backend->window->set_clipboard_text(text);
    };
    ctx->get()->PlatformIO.Platform_GetClipboardTextFn = [](::ImGuiContext* c) -> const char* {
        auto* backend = static_cast<ImGuiMerianWindowBackend*>(c->IO.BackendPlatformUserData);
        if (backend && backend->window)
            return backend->window->get_clipboard_text();
        return nullptr;
    };
    ctx->get()->PlatformIO.Platform_SetImeDataFn = [](::ImGuiContext* c, ImGuiViewport*,
                                                      ImGuiPlatformImeData* data) {
        auto* backend = static_cast<ImGuiMerianWindowBackend*>(c->IO.BackendPlatformUserData);
        if (!backend || !backend->window)
            return;
        const bool want = data->WantVisible || data->WantTextInput;
        if (!want) {
            if (backend->window->is_text_input_active())
                backend->window->stop_text_input();
            return;
        }
        backend->window->set_text_input_area(static_cast<int>(data->InputPos.x),
                                             static_cast<int>(data->InputPos.y), 1,
                                             static_cast<int>(data->InputLineHeight));
        if (!backend->window->is_text_input_active())
            backend->window->start_text_input();
    };
}

ImGuiMerianWindowBackend::~ImGuiMerianWindowBackend() {
    ::ImGuiContext* raw = ctx->get();
    if (raw->IO.BackendPlatformUserData == this) {
        raw->IO.BackendFlags &=
            ~(ImGuiBackendFlags_HasMouseCursors | ImGuiBackendFlags_HasSetMousePos);
        raw->PlatformIO.Platform_SetClipboardTextFn = nullptr;
        raw->PlatformIO.Platform_GetClipboardTextFn = nullptr;
        raw->PlatformIO.Platform_SetImeDataFn = nullptr;
    }
    if (window && window->is_text_input_active())
        window->stop_text_input();
}

void ImGuiMerianWindowBackend::new_frame(const float delta_time) {
    if (ctx->get_io().WantSetMousePos) {
        const ImVec2 p = ctx->get_io().MousePos;
        window->set_cursor_pos(static_cast<double>(p.x), static_cast<double>(p.y));
    }
    if (((ctx->get_io().ConfigFlags & ImGuiConfigFlags_NoMouseCursorChange) == 0) &&
        !window->is_mouse_grabbed()) {
        ctx->with_context([this] {
            const int imgui_cursor = ImGui::GetMouseCursor();
            const bool hide =
                (imgui_cursor == ImGuiMouseCursor_None) || ctx->get_io().MouseDrawCursor;
            window->set_cursor(
                cursor_shape_from_imgui(hide ? ImGuiMouseCursor_None : imgui_cursor));
        });
    }
    ImGuiMerianBackend::new_frame(delta_time);
}

} // namespace merian
