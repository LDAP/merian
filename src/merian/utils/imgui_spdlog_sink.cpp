#include "imgui.h"
#include <merian/utils/imgui_spdlog_sink.hpp>

namespace merian {

ImguiSpdlogSink::ImguiSpdlogSink(const uint32_t buffer_size_lines) : log_lines(buffer_size_lines) {}

// ------------------------------------------------

void ImguiSpdlogSink::sink_it_(const spdlog::details::log_msg& msg) {
    LogLine& line = log_lines[log_line_write_index];

    line.buf.clear();
    formatter_->format(msg, line.buf);
    line.level = msg.level;

    log_line_write_index++;
    if (log_line_write_index == log_lines.size()) {
        log_line_write_index = 0;
    }
}

void ImguiSpdlogSink::flush_() { /* no need to flush */ }

// ------------------------------------------------

// draws the log to imgui
void ImguiSpdlogSink::imgui_draw_log() {
    ImGui::Checkbox("auto scroll", &auto_scoll);
    ImGui::SameLine();
    ImGui::Checkbox("wrap", &wrap);
    if (ImGui::BeginCombo("log level", spdlog::level::to_string_view(log_level).data())) {
        for (int l = 0; l < spdlog::level::level_enum::n_levels; l++) {
            if (ImGui::Selectable(
                    spdlog::level::to_string_view(spdlog::level::level_enum(l)).data(),
                    l == log_level)) {
                log_level = spdlog::level::level_enum(l);
            }
        }
        ImGui::EndCombo();
    }

    ImGui::Separator();

    ImGui::BeginChild("SpdLogTextView", ImVec2(0, ImGui::GetFontSize() * 20), ImGuiChildFlags{},
                      wrap ? 0 : ImGuiWindowFlags_HorizontalScrollbar);

    uint32_t log_line_read_index = log_line_write_index % log_lines.size();
    for (uint32_t i = 0; i < log_lines.size(); i++, log_line_read_index++) {
        if (log_line_read_index == log_lines.size()) {
            log_line_read_index = 0;
        }
        LogLine& line = log_lines[log_line_read_index];
        if (line.level >= log_level && line.buf.size() > 0) {
            if (wrap) {
                ImGui::TextWrapped(line.buf.begin(), line.buf.end());
            } else {
                ImGui::TextUnformatted(line.buf.begin(), line.buf.end());
            }
        }
    }

    if (auto_scoll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);

    ImGui::EndChild();
}

// creates a new window and draws the log there
void ImguiSpdlogSink::imgui_draw_window() {
    if (ImGui::Begin("Merian Log")) {
        imgui_draw_log();
    }
    ImGui::End();
}

} // namespace merian
