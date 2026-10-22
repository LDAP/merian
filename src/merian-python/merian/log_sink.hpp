#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <spdlog/sinks/base_sink.h>

namespace merian::python {

// Buffers the newest records, up to capacity, until they are read; thread-safe.
class LogSink : public spdlog::sinks::base_sink<std::mutex> {
  public:
    struct Record {
        spdlog::level::level_enum level;
        spdlog::log_clock::time_point time;
        std::string logger_name;
        std::string message;
        std::optional<std::string> source_file;
        std::optional<int> source_line;
        std::optional<std::string> source_function;
    };

    // Joins the default logger, before other threads log.
    static void install();

    static std::shared_ptr<LogSink> attach(const std::size_t capacity,
                                           const spdlog::level::level_enum level);

    explicit LogSink(const std::size_t capacity);

    void detach();

    // Oldest first.
    std::vector<Record> get_new_records();

  protected:
    void sink_it_(const spdlog::details::log_msg& msg) override;

    void flush_() override;

  private:
    const std::size_t capacity;
    std::deque<Record> records;
};

} // namespace merian::python
