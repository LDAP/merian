#pragma once

#include <spdlog/sinks/base_sink.h>

namespace merian {

// Acts as a spdlog sink that can render to ImGui.
class ImguiSpdlogSink : public spdlog::sinks::base_sink<std::mutex> {

    using sink_t = spdlog::sinks::base_sink<std::mutex>;

public:
    ImguiSpdlogSink(const uint32_t buffer_size_lines = 512);

  protected:
    void sink_it_(const spdlog::details::log_msg& msg) override;

    void flush_() override;

  public:
    // ------------------------------------------------

    // draws the log to imgui
    void imgui_draw_log();

    // creates a new window and draws the log there
    void imgui_draw_window();

  private:
    struct LogLine {
        spdlog::memory_buf_t buf;
        spdlog::level::level_enum level;
    };

    std::vector<LogLine> log_lines;
    std::size_t log_line_write_index = 0;

    bool needs_scroll = false;
    bool auto_scoll = true;
    bool wrap = false;
    spdlog::level::level_enum log_level = spdlog::level::trace;
};

} // namespace merian
