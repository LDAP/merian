#include "merian-nodes/nodes/image_write/image_write.hpp"
#include "stb_image_write.h"

#include "merian/utils/defer.hpp"
#include "merian/vk/utils/blits.hpp"

#include <csignal>
#include <filesystem>

#include "fmt/args.h"

namespace merian {

#define FORMAT_PNG 0
#define FORMAT_JPG 1
#define FORMAT_HDR 2

static const std::unordered_map<uint32_t, std::string> FILE_EXTENSIONS = {
    {FORMAT_PNG, ".png"}, {FORMAT_JPG, ".jpg"}, {FORMAT_HDR, ".hdr"}};

ImageWrite::ImageWrite() {}

ImageWrite::~ImageWrite() {}

void ImageWrite::initialize(const ContextHandle& context,
                            const ResourceAllocatorHandle& allocator) {
    this->context = context;
    this->allocator = allocator;
}

std::vector<InputConnectorDescriptor> ImageWrite::describe_inputs() {
    return {{"src", con_src}};
}

void ImageWrite::record(const std::chrono::nanoseconds& current_graph_time) {
    record_enable = true;
    needs_rebuild |= rebuild_on_record;
    iteration = 1;
    last_record_time_millis = -std::numeric_limits<double>::infinity();
    last_frame_time_millis = 0;
    record_graph_time_point = current_graph_time;
    num_captures_since_record = 0;
    record_iteration_at_start = record_iteration;
    record_time_point.reset();

    if (callback_on_record && callback)
        callback();
}

ImageWrite::NodeStatusFlags ImageWrite::pre_process(const GraphRun& run,
                                                    [[maybe_unused]] const NodeIO& io) {
    // START TRIGGER
    if (!record_enable && (start_stop_record || (int64_t)run.get_iteration() == start_at_run)) {
        record(run.get_elapsed_duration());
        start_stop_record = false;
        io.send_event("start");
    }

    const std::chrono::nanoseconds time_since_record =
        run.get_elapsed_duration() - record_graph_time_point;

    // STOP TRIGGER
    if (record_enable &&
        (start_stop_record || stop_at_run == (int64_t)run.get_iteration() ||
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

    // SIGTERM TRIGGER
    if (exit_at_run == (int64_t)run.get_iteration() || exit_at_iteration == iteration) {
        raise(SIGTERM);
    }
    if (exit_after_seconds >= 0 && to_seconds(time_since_record) >= exit_after_seconds) {
        raise(SIGTERM);
    }

    if (needs_rebuild) {
        needs_rebuild = false;
        return NodeStatusFlagBits::NEEDS_RECONNECT;
    }

    return {};
};

void ImageWrite::process(GraphRun& run,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) {

    //--------- Make sure we always increase the iteration counter
    defer {
        iteration++;
    };

    const CommandBufferHandle& cmd = run.get_cmd();

    const std::chrono::nanoseconds system_time_since_record = record_time_point.duration();
    const std::chrono::nanoseconds& graph_time = run.get_elapsed_duration();
    const std::chrono::nanoseconds graph_time_since_record = graph_time - record_graph_time_point;

    assert(time_reference == 0 || time_reference == 1);
    const std::chrono::nanoseconds time_since_record =
        time_reference == 0 ? system_time_since_record : graph_time_since_record;

    //--------- RECORD TRIGGER
    // RECORD TRIGGER 0: Iteration
    record_next |= record_enable && (trigger == 0) && record_iteration == iteration;

    // RECORD TRIGGER 1: Frametime
    const double time_millis = to_milliseconds(time_since_record);
    const double optimal_timing = last_record_time_millis + record_frametime_millis;
    if (record_enable && (trigger == 1) && last_frame_time_millis <= 0) {
        record_next = true;
    } else {
        // estimate how long a frame takes and reduce stutter
        const double frametime_millis = time_millis - last_frame_time_millis;

        // am I this time closer to the optimal point or next frame?
        if (record_enable && (trigger == 1) &&
            std::abs(time_millis - optimal_timing) <
                std::abs(time_millis + frametime_millis - optimal_timing)) {
            record_next = true;
            undersampling = (frametime_millis > record_frametime_millis);
            if (undersampling)
                SPDLOG_WARN("undersampling, video may stutter");
        }
    }

    last_frame_time_millis = time_millis;

    // CHECK PATH
    const ImageHandle src = io[con_src];
    vk::Extent3D scaled = max(src->get_extent() * scale, {1, 1, 1});
    fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
    get_format_args([&](const auto& arg) { arg_store.push_back(arg); }, src->get_extent(), scaled,
                    run.get_iteration(), graph_time_since_record, graph_time,
                    system_time_since_record);
    std::filesystem::path path;
    try {
        if (filename_format.empty()) {
            throw fmt::format_error{"empty filename"};
        }
        path = std::filesystem::absolute(fmt::vformat(this->filename_format, arg_store) +
                                         FILE_EXTENSIONS.at(this->format));
    } catch (const std::exception& e) {
        // catch std::filesystem::filesystem_error and fmt::format_error
        record_enable = false;
        record_next = false;
        SPDLOG_ERROR(e.what());
        return;
    }

    if (!record_next)
        return;
    // needs correction else we might take more pictures if the
    // record framerate is slightly below the actual framerate
    last_record_time_millis = std::max(time_millis, optimal_timing);

    // RECORD FRAME

    const vk::Format format =
        this->format == FORMAT_HDR ? vk::Format::eR32G32B32A32Sfloat : vk::Format::eR8G8B8A8Srgb;
    const vk::FormatProperties format_properties =
        context->get_physical_device()->get_physical_device().getFormatProperties(format);

    vk::ImageCreateInfo linear_info{
        {},
        vk::ImageType::e2D,
        format,
        scaled,
        1,
        1,
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eLinear,
        vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive,
        {},
        {},
        vk::ImageLayout::eUndefined,
    };
    ImageHandle linear_image =
        allocator->create_image(linear_info, MemoryMappingType::HOST_ACCESS_RANDOM);
    cmd->barrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                 linear_image->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                                       vk::AccessFlagBits::eTransferWrite));

    if (format_properties.linearTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst) {
        // blit directly onto the linear image
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "blit to linear image");
        cmd_blit_stretch(cmd, src, src->get_current_layout(), src->get_extent(), linear_image,
                         vk::ImageLayout::eTransferDstOptimal, linear_image->get_extent());

    } else {
        // cannot blit directly to the linear image with the desired format
        // therefore blit first onto a optimal tiled image and then copy to linear tiled image.
        vk::ImageCreateInfo intermediate_info{
            {},
            vk::ImageType::e2D,
            format,
            scaled,
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc,
            vk::SharingMode::eExclusive,
            {},
            {},
            vk::ImageLayout::eUndefined,
        };
        ImageHandle intermediate_image = allocator->create_image(intermediate_info);

        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "blit to optimal tiled image");

            cmd->barrier(vk::PipelineStageFlagBits::eTopOfPipe,
                         vk::PipelineStageFlagBits::eTransfer,
                         intermediate_image->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                                                     vk::AccessFlagBits::eTransferWrite));
            cmd_blit_stretch(cmd, src, src->get_current_layout(), src->get_extent(),
                             intermediate_image, vk::ImageLayout::eTransferDstOptimal,
                             intermediate_image->get_extent());
            cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                         intermediate_image->barrier(vk::ImageLayout::eTransferSrcOptimal,
                                                     vk::AccessFlagBits::eTransferWrite,
                                                     vk::AccessFlagBits::eTransferRead));
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "copy to linear image");
            cmd->copy(intermediate_image, linear_image);
        }
    }
    cmd->barrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                 linear_image->barrier(vk::ImageLayout::eGeneral,
                                       vk::AccessFlagBits::eTransferWrite,
                                       vk::AccessFlagBits::eHostRead));

    std::filesystem::create_directories(path.parent_path());
    const std::string tmp_filename =
        (path.parent_path() / (".interm_" + path.filename().string())).string();

    std::function<void()> write_task = ([this, linear_image, path, tmp_filename]() {
        float* mem = linear_image->get_memory()->map_as<float>();

        switch (this->format) {
        case FORMAT_PNG: {
            stbi_write_png(tmp_filename.c_str(), linear_image->get_extent().width,
                           linear_image->get_extent().height, 4, mem,
                           linear_image->get_extent().width * 4);
            break;
        }
        case FORMAT_JPG: {
            stbi_write_jpg(tmp_filename.c_str(), linear_image->get_extent().width,
                           linear_image->get_extent().height, 4, mem, 100);
            break;
        }
        case FORMAT_HDR: {
            stbi_write_hdr(tmp_filename.c_str(), linear_image->get_extent().width,
                           linear_image->get_extent().height, 4, mem);
            break;
        }
        default:
            throw std::runtime_error{"unsupported format."};
        }

        try {
            std::filesystem::rename(tmp_filename, path);
        } catch (std::filesystem::filesystem_error const&) {
            SPDLOG_WARN("rename failed! Falling back to copy...");
            std::filesystem::copy(tmp_filename, path);
            std::filesystem::remove(tmp_filename);
        }

        SPDLOG_INFO("wrote image to {}", path.string());

        linear_image->get_memory()->unmap();
        return;
    });

    run.sync_to_cpu(std::move(write_task));

    if (rebuild_after_capture)
        run.request_reconnect();
    if (callback_after_capture && callback)
        callback();
    io.send_event("capture");
    record_next = false;
    num_captures_since_record++;
    num_captures_since_init++;

    record_iteration *= record_enable ? it_power : 1;
    record_iteration += record_enable ? it_offset : 0;
}

ImageWrite::NodeStatusFlags ImageWrite::properties([[maybe_unused]] Properties& config) {
    config.st_separate("General");
    config.config_options("format", format, {"PNG", "JPG", "HDR"}, Properties::OptionsStyle::COMBO);
    std::ignore = config.config_text("filename", filename_format, false,
                                     "Provide a format string for the path.");
    std::vector<std::string> variables;
    get_format_args([&](const auto& arg) { variables.push_back(arg.name); }, {1920, 1080, 1},
                    {1920, 1080, 1}, 1, 1000ns, 1000ns, 1000ns);
    fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
    get_format_args([&](const auto& arg) { arg_store.push_back(arg); }, {1920, 1080, 1},
                    {1920, 1080, 1}, 1, 1000ns, 1000ns, 1000ns);

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
    record_next = config.config_bool("record_next");

    config.st_separate("Multiple");
    config.output_text(fmt::format("current iteration: {}\nundersampling: {}",
                                   record_enable ? fmt::to_string(iteration) : "stopped",
                                   undersampling));
    bool prop_record_enable = record_enable;
    start_stop_record = config.config_bool("enable", prop_record_enable);
    config.st_separate();

    config.config_options("trigger", trigger, {"iteration", "frametime"},
                          Properties::OptionsStyle::COMBO);
    if (trigger == 0) {
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
    if (trigger == 1) {
        config.config_options("time reference", time_reference, {"system", "graph"},
                              Properties::OptionsStyle::COMBO);
        config.config_float("framerate", record_framerate, "", 0.01);
        record_frametime_millis = 1000 / record_framerate;
        config.config_float("frametime", record_frametime_millis, "", 0.01);
        record_framerate = 1000 / record_frametime_millis;
    }
    config.st_separate();
    if (config.st_begin_child("advanced", "Advanced")) {
        config.config_percent("scale", scale);
        config.st_separate();

        config.config_bool("rebuild after capture", rebuild_after_capture,
                           "forces a graph rebuild after every capture");
        config.config_bool("rebuild on record", rebuild_on_record,
                           "Rebuilds when recording starts");
        config.config_bool("callback after capture", callback_after_capture,
                           "calls the on_record callback after every capture");
        config.config_bool("callback on record", callback_on_record,
                           "calls the callback when the recording starts");
        config.st_separate();
        config.config_int(
            "start at run", start_at_run,
            "The specified run starts recording and resets the iteration and calls the "
            "configured callback and forces a rebuild if enabled.");
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
        config.config_int(
            "exit at run", exit_at_run,
            "Raises SIGTERM at the specified run. -1 to disable. Add a signal handler to "
            "shut down properly and not corrupt the images.");
        config.config_int("exit at iteration", exit_at_iteration,
                          "Raises SIGTERM at the specified iteration. -1 to disable. Add a signal "
                          "handler to shut down properly and not corrupt the images.");
        config.config_float(
            "exit after seconds", exit_after_seconds,
            "Raises SIGTERM after the specified seconds have passed. -1 to disable. Add a signal "
            "handler to shut down properly and not corrupt the images.");
        config.st_end_child();
    }
    config.st_separate();
    config.output_text(
        "Hint: convert to video with ffmpeg -framerate <framerate> -pattern_type glob -i "
        "'*.jpg' -level 3.0 -pix_fmt yuv420p out.mp4");
    return {};
}

void ImageWrite::set_callback(const std::function<void()>& callback) {
    this->callback = callback;
}

} // namespace merian
