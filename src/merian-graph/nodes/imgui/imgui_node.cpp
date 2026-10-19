#include "merian-graph/nodes/imgui/imgui_node.hpp"

#include "merian/utils/properties_imgui.hpp"
#include "merian/vk/imgui/imgui_merian_window_backend.hpp"
#include "merian/vk/imgui/imgui_theme.hpp"

#include <imgui.h>

namespace merian {

void ImGuiNode::initialize(const ContextHandle& context, const ResourceAllocatorHandle& allocator) {
    imgui_ctx = std::make_shared<ImGuiContext>(theme);
    imgui_renderer = std::make_shared<ImGuiRenderer>(context, allocator, imgui_ctx);
    imgui_backend = std::make_shared<ImGuiMerianBackend>(imgui_ctx);
}

std::vector<InputConnectorDescriptor> ImGuiNode::describe_inputs() {
    return {{.name = "canvas",
             .connector = con_canvas,
             .access = ConnectorAccess::color_attachment,
             .in_place = true},
            {.name = "window", .connector = con_window}};
}

[[nodiscard]] ImGuiNode::NodeStatusFlags
ImGuiNode::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {
    const TextureHandle& canvas = io[con_canvas].get_texture();
    if (!canvas) {
        return {};
    }

    const std::shared_ptr<Window>& window = io[con_window];
    if (window && window != current_window) {
        imgui_backend = std::make_shared<ImGuiMerianWindowBackend>(imgui_ctx, window);
        current_window = window;
    }

    imgui_backend->new_frame(static_cast<float>(frametime.seconds()));
    frametime.reset();

    const double dt_ms = info.get_time_delta() * 1000.0;
    const std::string title =
        fmt::format("merian ({:.2f} ms, {:.1f} fps) Frame {}###{}", dt_ms,
                    dt_ms > 0.0 ? 1000.0 / dt_ms : 0.0, info.get_iteration(), imgui_event);
    imgui_ctx->with_context([&] {
        // Listeners that open windows of their own draw them while this one is collapsed too
        ImGui::Begin(title.c_str());
        ImGuiProperties props;
        io.send_event(imgui_event, static_cast<Properties*>(&props));
        ImGui::End();
    });

    imgui_renderer->render(submission.get_cmd(), canvas->get_view());
    return {};
}

ImGuiNode::NodeStatusFlags ImGuiNode::properties(Properties& config) {
    static_cast<void>(config.config_text("imgui event", imgui_event));
    if (config.config_enum("theme", theme, Properties::OptionsStyle::COMBO) && imgui_ctx) {
        imgui_ctx->with_context([&] { apply_imgui_theme(ImGui::GetStyle(), theme); });
    }
    return {};
}

} // namespace merian
