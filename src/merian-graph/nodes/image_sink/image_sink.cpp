#include "merian-graph/nodes/image_sink/image_sink.hpp"

#include "merian/io/image_io.hpp"
#include "merian/utils/defer.hpp"
#include "merian/vk/utils/image_export.hpp"

#include <filesystem>

#include "fmt/args.h"

namespace merian {

using namespace std::literals::chrono_literals;

std::vector<InputConnectorDescriptor> ImageSink::describe_inputs() {
    return {{"src", con_src, ConnectorAccess::transfer_src}};
}

void ImageSink::record(const std::chrono::nanoseconds& current_graph_time) {
    record_enable = true;
    needs_rebuild |= rebuild_on_record;
    iteration = 1;
    last_record_time_millis = -std::numeric_limits<double>::infinity();
    last_frame_time_millis = 0;
    record_graph_time_point = current_graph_time;
    num_captures_since_record = 0;
    record_iteration_at_start = record_iteration;
    record_time_point.reset();
}

bool ImageSink::provides_time() const {
    return time_provider != TimeProviderMode::OFF;
}

std::optional<std::chrono::nanoseconds> ImageSink::provide_time() {
    if (time_provider == TimeProviderMode::WHEN_ACTIVE && !record_enable) {
        // the timeline starts at zero once this node starts providing
        provided_frames = 0;
        return std::nullopt;
    }
    if (provided_frames == 0) {
        provided_rate = record_rate;
    }
    return provided_rate.frame_time(provided_frames++);
}

ImageSink::NodeStatusFlags ImageSink::pre_process(const NodeIO& io, const NodeProcessInfo& info) {
    // START TRIGGER
    if (!record_enable &&
        (start_stop_record || static_cast<int64_t>(info.get_iteration()) == start_at_run)) {
        record(info.get_elapsed_duration());
        start_stop_record = false;
        io.send_event("start");
    }

    // the graph clock jumps back when a provider takes it over, and on a seek or a loop
    record_graph_time_point = std::min(record_graph_time_point, info.get_elapsed_duration());
    const std::chrono::nanoseconds time_since_record =
        info.get_elapsed_duration() - record_graph_time_point;

    // STOP TRIGGER
    if (record_enable &&
        (start_stop_record || stop_at_run == static_cast<int64_t>(info.get_iteration()) ||
         (stop_after_iteration >= 0 && stop_after_iteration < iteration) ||
         stop_after_num_captures_since_record == num_captures_since_record ||
         (stop_after_seconds >= 0 && to_seconds(time_since_record) >= stop_after_seconds))) {
        record_enable = false;
        start_stop_record = false;
        num_captures_since_record = 0;
        iteration = 1;
        if (reset_record_iteration_at_stop) {
            record_iteration = record_iteration_at_start;
        }
        io.send_event("stop");
    }

    // EXIT TRIGGER
    if (exit_at_run == static_cast<int64_t>(info.get_iteration()) ||
        exit_at_iteration == iteration ||
        (exit_after_seconds >= 0 && to_seconds(time_since_record) >= exit_after_seconds)) {
        io.send_event("end");
    }

    if (needs_rebuild) {
        needs_rebuild = false;
        return NodeStatusFlagBits::NEEDS_RECONNECT;
    }

    return {};
};

[[nodiscard]] ImageSink::NodeStatusFlags
ImageSink::process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) {

    //--------- Make sure we always increase the iteration counter
    defer {
        iteration++;
    };

    const std::chrono::nanoseconds system_time_since_record = record_time_point.duration();
    const std::chrono::nanoseconds& graph_time = info.get_elapsed_duration();
    const std::chrono::nanoseconds graph_time_since_record = graph_time - record_graph_time_point;

    const std::chrono::nanoseconds time_since_record =
        time_reference == ImageSinkTimeReference::SYSTEM ? system_time_since_record
                                                         : graph_time_since_record;

    //--------- RECORD TRIGGER
    // RECORD TRIGGER 0: Iteration
    record_next |=
        record_enable && (trigger == ImageSinkTrigger::ITERATION) && record_iteration == iteration;

    // RECORD TRIGGER 1: End of run
    record_next |=
        record_enable && trigger == ImageSinkTrigger::END_OF_RUN && info.is_last_iteration();

    // RECORD TRIGGER 2: Frametime
    const double time_millis = to_milliseconds(time_since_record);
    const double optimal_timing = last_record_time_millis + record_rate.get_frametime_millis();
    if (record_enable && (trigger == ImageSinkTrigger::FRAMETIME) && last_frame_time_millis <= 0) {
        record_next = true;
    } else {
        // estimate how long a frame takes and reduce stutter
        const double frametime_millis = time_millis - last_frame_time_millis;

        // am I this time closer to the optimal point or next frame?
        if (record_enable && (trigger == ImageSinkTrigger::FRAMETIME) &&
            std::abs(time_millis - optimal_timing) <
                std::abs(time_millis + frametime_millis - optimal_timing)) {
            record_next = true;
            undersampling = (frametime_millis > record_rate.get_frametime_millis());
            if (undersampling)
                SPDLOG_WARN("undersampling, video may stutter");
        }
    }

    last_frame_time_millis = time_millis;

    // CHECK PATH
    const ImageHandle src = io[con_src];
    const vk::Extent3D scaled = max(src->get_extent() * scale, {1, 1, 1});
    fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
    get_format_args([&](const auto& arg) { arg_store.push_back(arg); }, src->get_extent(), scaled,
                    info.get_iteration(), info.get_total_iteration(), graph_time_since_record,
                    graph_time, system_time_since_record);
    std::filesystem::path path;
    try {
        if (filename_format.empty()) {
            throw fmt::format_error{"empty filename"};
        }
        path = std::filesystem::absolute(
            fmt::vformat(this->filename_format, arg_store) +
            image_format_extension(IMAGE_EXPORT_FORMATS.at(this->format)));
    } catch (const std::exception& e) {
        // catch std::filesystem::filesystem_error and fmt::format_error
        record_enable = false;
        record_next = false;
        SPDLOG_ERROR(e.what());
        return {};
    }

    if (!record_next)
        return {};
    // needs correction else we might take more pictures if the
    // record framerate is slightly below the actual framerate
    last_record_time_millis = std::max(time_millis, optimal_timing);

    // RECORD FRAME
    image_export(io.get_allocator(), submission, info.get_profiler(), src,
                 vk::Extent2D{scaled.width, scaled.height}, path,
                 IMAGE_EXPORT_FORMATS.at(this->format),
                 embed_metadata ? info.get_metadata() : ImageMetadata{}, !keep_alpha);

    NodeStatusFlags flags{};
    if (rebuild_after_capture)
        flags |= NodeStatusFlagBits::NEEDS_RECONNECT;
    io.send_event("capture");
    record_next = false;
    num_captures_since_record++;
    num_captures_since_init++;

    record_iteration *= record_enable ? it_power : 1;
    record_iteration += record_enable ? it_offset : 0;
    return flags;
}

ImageSink::NodeStatusFlags ImageSink::properties(Properties& config) {
    config.st_separate("General");
    config.config_options("format", format, image_export_format_names(),
                          Properties::OptionsStyle::COMBO);
    std::ignore = config.config_text("filename", filename_format, false,
                                     "Provide a format string for the path.");
    std::vector<std::string> variables;
    get_format_args([&](const auto& arg) { variables.push_back(arg.name); }, {1920, 1080, 1},
                    {1920, 1080, 1}, 1, 1, 1000ns, 1000ns, 1000ns);
    fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
    get_format_args([&](const auto& arg) { arg_store.push_back(arg); }, {1920, 1080, 1},
                    {1920, 1080, 1}, 1, 1, 1000ns, 1000ns, 1000ns);

    std::filesystem::path abs_path;
    try {
        abs_path = std::filesystem::absolute(fmt::vformat(filename_format, arg_store)).string();
    } catch (const std::exception& e) {
        // catch std::filesystem::filesystem_error and fmt::format_error
        abs_path.clear();
    }
    config.output_text(
        fmt::format("abs path: {}", abs_path.empty() ? "<invalid>" : abs_path.string()));
    config.output_text(fmt::format("variables: {}", fmt::join(variables, ", ")));

    config.st_separate("Single");
    record_next = config.config_bool("record next");

    config.st_separate("Multiple");
    config.output_text(fmt::format("current iteration: {}\nundersampling: {}",
                                   record_enable ? fmt::to_string(iteration) : "stopped",
                                   undersampling));
    bool prop_record_enable = record_enable;
    start_stop_record = config.config_bool("enable", prop_record_enable);
    config.config_enum("time provider", time_provider, Properties::OptionsStyle::COMBO,
                       TIME_PROVIDER_DESCRIPTION);
    config.st_separate();

    config.config_enum("trigger", trigger, Properties::OptionsStyle::COMBO,
                       "end of run: the last iteration of a run given an iteration count.");
    if (trigger == ImageSinkTrigger::ITERATION) {
        config.config_int(
            "iteration", record_iteration,
            "Save the result of of the the specified iteration. Iterations are 1-indexed.");
        record_iteration = std::max(record_iteration, 0);

        config.config_int("iteration power", it_power,
                          "Multiplies the iteration specifier with this value after every capture");
        config.config_int("iteration offset", it_offset,
                          "Adds this value to the iteration specifier after every capture. (After "
                          "applying the power).");
        config.config_bool(
            "reset iteration at stop", reset_record_iteration_at_stop,
            "resets the record iteration to the value it had when recording started.");
        config.output_text("note: Iterations are 1-indexed");
    }
    if (trigger == ImageSinkTrigger::FRAMETIME) {
        config.config_enum("time reference", time_reference, Properties::OptionsStyle::COMBO);
    }
    if (trigger == ImageSinkTrigger::FRAMETIME || time_provider != TimeProviderMode::OFF) {
        record_rate.properties(config);
    }
    config.st_separate();
    if (config.st_begin_child("advanced", "Advanced")) {
        config.config_percent("scale", scale);
        config.config_bool("keep alpha", keep_alpha,
                           "Writes the graph's alpha channel into PNG/TGA instead of forcing "
                           "opaque; viewers composite it.");
        config.config_bool("embed metadata", embed_metadata,
                           "embeds the merian version and the graph config into the file (PNG, "
                           "JPG, HDR)");
        config.st_separate();

        config.config_bool("rebuild after capture", rebuild_after_capture,
                           "forces a graph rebuild after every capture");
        config.config_bool("rebuild on record", rebuild_on_record,
                           "Rebuilds when recording starts");
        config.st_separate();
        config.config_int("start at run", start_at_run,
                          "The specified run starts recording, resets the iteration and forces a "
                          "rebuild if enabled.");
        config.st_separate();
        config.config_int("stop at run", stop_at_run,
                          "Stops recording at the specified run (before capture). -1 to disable.");
        config.config_int(
            "stop after iteration", stop_after_iteration,
            "Stops recording after the specified iteration (after capture). -1 to disable.");
        config.config_int("stop after number captures", stop_after_num_captures_since_record,
                          "stops recording after the specified number of images have been captured "
                          "since recording started.");
        config.config_float(
            "stop after seconds", stop_after_seconds,
            "Stops recording after the specified seconds have passed. -1 to dissable.");
        config.config_int("exit at run", exit_at_run,
                          "Sends \"end\" at the specified run. -1 to disable.");
        config.config_int("exit at iteration", exit_at_iteration,
                          "Sends \"end\" at the specified iteration. -1 to disable.");
        config.config_float(
            "exit after seconds", exit_after_seconds,
            "Sends \"end\" after the specified seconds have passed. -1 to disable.");
        config.st_end_child();
    }
    config.st_separate();
    config.output_text(
        "Hint: convert to video with ffmpeg -framerate <framerate> -pattern_type glob -i "
        "'*.jpg' -level 3.0 -pix_fmt yuv420p out.mp4");
    return {};
}

} // namespace merian
