#include "log_sink.hpp"

#include <algorithm>
#include <iterator>
#include <spdlog/sinks/dist_sink.h>
#include <spdlog/spdlog.h>

namespace merian::python {

namespace {

class Forwarder : public spdlog::sinks::sink {
  public:
    explicit Forwarder(const std::weak_ptr<LogSink>& target) : target(target) {}

    void log(const spdlog::details::log_msg& msg) override {
        if (const std::shared_ptr<LogSink> sink = target.lock();
            sink && sink->should_log(msg.level)) {
            sink->log(msg);
        }
    }

    void flush() override {}

    void set_pattern([[maybe_unused]] const std::string& pattern) override {}

    void set_formatter([[maybe_unused]] std::unique_ptr<spdlog::formatter> formatter) override {}

    const std::weak_ptr<LogSink> target;
};

struct Attached {
    std::mutex mutex;
    std::vector<std::shared_ptr<Forwarder>> forwarders;
    const std::shared_ptr<spdlog::sinks::dist_sink_mt> sinks =
        std::make_shared<spdlog::sinks::dist_sink_mt>();

    // With mutex held.
    void publish() {
        std::erase_if(forwarders, [](const std::shared_ptr<Forwarder>& forwarder) {
            return forwarder->target.expired();
        });
        sinks->set_sinks({forwarders.begin(), forwarders.end()});
    }
};

Attached& attached() {
    static Attached attached;
    return attached;
}

} // namespace

void LogSink::install() {
    spdlog::default_logger()->sinks().push_back(attached().sinks);
}

std::shared_ptr<LogSink> LogSink::attach(const std::size_t capacity,
                                         const spdlog::level::level_enum level) {
    const auto sink = std::make_shared<LogSink>(capacity);
    sink->set_level(level);
    Attached& all = attached();
    const std::lock_guard lock(all.mutex);
    all.forwarders.push_back(std::make_shared<Forwarder>(sink));
    all.publish();
    return sink;
}

LogSink::LogSink(const std::size_t capacity) : capacity(capacity) {}

void LogSink::detach() {
    Attached& all = attached();
    const std::lock_guard lock(all.mutex);
    std::erase_if(all.forwarders, [&](const std::shared_ptr<Forwarder>& forwarder) {
        return forwarder->target.lock().get() == this;
    });
    all.publish();
}

std::vector<LogSink::Record> LogSink::get_new_records() {
    const std::lock_guard lock(mutex_);
    std::vector<Record> new_records(std::make_move_iterator(records.begin()),
                                    std::make_move_iterator(records.end()));
    records.clear();
    return new_records;
}

void LogSink::sink_it_(const spdlog::details::log_msg& msg) {
    if (capacity == 0) {
        return;
    }
    if (records.size() == capacity) {
        records.pop_front();
    }
    Record& record = records.emplace_back(Record{
        .level = msg.level,
        .time = msg.time,
        .logger_name = std::string(msg.logger_name.data(), msg.logger_name.size()),
        .message = std::string(msg.payload.data(), msg.payload.size()),
    });
    if (msg.source.empty()) {
        return;
    }
    record.source_line = msg.source.line;
    if (msg.source.filename != nullptr) {
        record.source_file = msg.source.filename;
    }
    if (msg.source.funcname != nullptr) {
        record.source_function = msg.source.funcname;
    }
}

void LogSink::flush_() {}

} // namespace merian::python
