#include "image_write.hpp"
#include "ext/stb_image_write.h"

#include "merian/utils/defer.hpp"
#include "merian/vk/utils/blits.hpp"

#include <csignal>
#include <filesystem>

#include "fmt/args.h"

namespace merian_nodes {

#define FORMAT_PNG 0
#define FORMAT_JPG 1
#define FORMAT_HDR 2

static std::unordered_map<uint32_t, std::string> FILE_EXTENSIONS = {
    {FORMAT_PNG, ".png"}, {FORMAT_JPG, ".jpg"}, {FORMAT_HDR, ".hdr"}};

ImageWrite::ImageWrite(const ContextHandle context,
                       const ResourceAllocatorHandle allocator,
                       const std::string& filename_format)
    : Node(), context(context), allocator(allocator), filename_format(filename_format), buf(1024) {
    assert(filename_format.size() < buf.size());
    std::copy(filename_format.begin(), filename_format.end(), buf.begin());
}

ImageWrite::~ImageWrite() {}

std::vector<InputConnectorHandle> ImageWrite::describe_inputs() {
    return {con_src};
}

void ImageWrite::record() {
    record_enable = true;
    needs_rebuild |= rebuild_on_record;
    this->iteration = 1;
    estimated_frametime_millis = 0;
    last_record_time_millis = -std::numeric_limits<double>::infinity();
    last_frame_time_millis = 0;
    time_since_record.reset();

    if (callback_on_record && callback)
        callback();
}

ImageWrite::NodeStatusFlags ImageWrite::pre_process(GraphRun& run,
                                                    [[maybe_unused]] const NodeIO& io) {
    if (!record_enable && ((int64_t)run.get_iteration() == enable_run)) {
        record();
    }
    if (needs_rebuild) {
        needs_rebuild = false;
        return NodeStatusFlagBits::NEEDS_RECONNECT;
    }
    return {};
};

void ImageWrite::process(GraphRun& run,
                         const vk::CommandBuffer& cmd,
                         [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                         const NodeIO& io) {

    //--------- Make sure we always increase the iteration counter
    defer {
        iteration++;
    };

    //--------- STOP TRIGGER
    if (stop_run == (int64_t)run.get_iteration() || stop_iteration == iteration) {
        record_enable = false;
    }
    if (exit_run == (int64_t)run.get_iteration() || exit_iteration == iteration) {
        raise(SIGTERM);
    }
    if (stop_after_seconds >= 0 && time_since_record.seconds() >= stop_after_seconds) {
        record_enable = false;
    }
    if (exit_after_seconds >= 0 && time_since_record.seconds() >= exit_after_seconds) {
        raise(SIGTERM);
    }

    //--------- RECORD TRIGGER
    // RECORD TRIGGER 0: Iteration
    record_next |= record_enable && (trigger == 0) && record_iteration == iteration;

    // RECORD TRIGGER 1: Frametime
    const double time_millis = time_since_record.millis();
    const double optimal_timing = last_record_time_millis + record_frametime_millis;
    if (record_enable && (trigger == 1) && last_frame_time_millis <= 0) {
        record_next = true;
    } else {
        // estimate how long a frame takes and reduce stutter
        const double frametime_millis = time_millis - last_frame_time_millis;

        if (estimated_frametime_millis == 0)
            estimated_frametime_millis = frametime_millis;
        else
            estimated_frametime_millis = estimated_frametime_millis * 0.9 + frametime_millis * 0.1;

        // am I this time closer to the optimal point or next frame?
        if (record_enable && (trigger == 1) &&
            std::abs(time_millis - optimal_timing) <
                std::abs(time_millis + estimated_frametime_millis - optimal_timing)) {
            record_next = true;
            if ((undersampling = (frametime_millis > record_frametime_millis)))
                SPDLOG_WARN("undersampling, video may stutter");
        }
    }

    last_frame_time_millis = time_millis;

    // CHECK PATH
    const ImageHandle src = io[con_src];
    vk::Extent3D scaled = max(multiply(src->get_extent(), scale), {1, 1, 1});
    fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
    get_format_args([&](const auto& arg) { arg_store.push_back(arg); }, scaled,
                    run.get_iteration());
    std::filesystem::path path;
    try {
        if (filename_format.empty()) {
            throw fmt::format_error{"empty filename"};
        }
        path = std::filesystem::absolute(fmt::vformat(this->filename_format, arg_store) +
                                         FILE_EXTENSIONS[this->format]);
    } catch (const fmt::format_error& e) {
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
        context->physical_device.physical_device.getFormatProperties(format);

    FrameData& frame_data = io.frame_data<FrameData>();
    frame_data.intermediate_image.reset();

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
        allocator->createImage(linear_info, MemoryMappingType::HOST_ACCESS_RANDOM);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                        {}, {}, {},
                        linear_image->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                                              vk::AccessFlagBits::eTransferWrite));

    if (format_properties.linearTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst) {
        // blit directly onto the linear image
        MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "blit to linear image");
        cmd_blit_stretch(cmd, *src, src->get_current_layout(), src->get_extent(), *linear_image,
                         vk::ImageLayout::eTransferDstOptimal, linear_image->get_extent());

    } else {
        // cannot blit directly to the linear image with the desired format
        // therefore blit first onto a optimal tiled image and then copy to linear tiled image.
        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "blit to optimal tiled image");
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
            ImageHandle intermediate_image = allocator->createImage(intermediate_info);
            frame_data.intermediate_image = intermediate_image;

            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                                intermediate_image->barrier(vk::ImageLayout::eTransferDstOptimal,
                                                            {},
                                                            vk::AccessFlagBits::eTransferWrite));
            cmd_blit_stretch(cmd, *src, src->get_current_layout(), src->get_extent(),
                             *intermediate_image, vk::ImageLayout::eTransferDstOptimal,
                             intermediate_image->get_extent());
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                                intermediate_image->barrier(vk::ImageLayout::eTransferSrcOptimal,
                                                            vk::AccessFlagBits::eTransferWrite,
                                                            vk::AccessFlagBits::eTransferRead));
        }
        {
            MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "copy to linear image");
            cmd.copyImage(*frame_data.intermediate_image.value(),
                          frame_data.intermediate_image.value()->get_current_layout(),
                          *linear_image, linear_image->get_current_layout(),
                          vk::ImageCopy(first_layer(), {}, first_layer(), {},
                                        frame_data.intermediate_image.value()->get_extent()));
        }
    }
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, {}, {},
        linear_image->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eTransferWrite,
                              vk::AccessFlagBits::eHostRead));

    TimelineSemaphoreHandle image_ready = std::make_shared<TimelineSemaphore>(context, 0);
    run.add_signal_semaphore(image_ready, 1);

    std::unique_lock lk(mutex_concurrent);
    cv_concurrent.wait(lk, [&] { return concurrent_tasks < max_concurrent_tasks; });
    concurrent_tasks++;
    lk.unlock();

    std::filesystem::create_directories(path.parent_path());
    const std::string tmp_filename =
        (path.parent_path() / (".interm_" + path.filename().string())).string();

    const std::function<void()> write_task =
        ([this, image_ready, linear_image, path, tmp_filename]() {
            image_ready->wait(1);
            void* mem = linear_image->get_memory()->map();

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
                               linear_image->get_extent().height, 4, static_cast<float*>(mem));
                break;
            }
            }

            try {
                std::filesystem::rename(tmp_filename, path);
            } catch (std::filesystem::filesystem_error const&) {
                SPDLOG_WARN("rename failed! Falling back to copy...");
                std::filesystem::copy(tmp_filename, path);
                std::filesystem::remove(tmp_filename);
            }

            linear_image->get_memory()->unmap();
            std::unique_lock lk(mutex_concurrent);
            concurrent_tasks--;
            lk.unlock();
            cv_concurrent.notify_all();
            return;
        });

    context->thread_pool.submit(write_task);

    if (rebuild_after_capture)
        run.request_reconnect();
    if (callback_after_capture && callback)
        callback();
    record_next = false;

    record_iteration *= record_enable ? it_power : 1;
    record_iteration += record_enable ? it_offset : 0;
}

ImageWrite::NodeStatusFlags ImageWrite::properties([[maybe_unused]] Properties& config) {
    config.st_separate("General");
    config.config_options("format", format, {"PNG", "JPG", "HDR"}, Properties::OptionsStyle::COMBO);
    config.config_bool("rebuild after capture", rebuild_after_capture,
                       "forces a graph rebuild after every capture");
    if (config.config_text("filename", buf.size(), buf.data(), false,
                           "Provide a format string for the path. Supported variables are: "
                           "record_iteration, run_iteration, image_index, width, height")) {
        filename_format = buf.data();
    }
    std::vector<std::string> variables;
    get_format_args([&](const auto& arg) { variables.push_back(arg.name); }, {1920, 1080, 1}, 1);
    fmt::dynamic_format_arg_store<fmt::format_context> arg_store;
    get_format_args([&](const auto& arg) { arg_store.push_back(arg); }, {1920, 1080, 1}, 1);

    std::filesystem::path abs_path;
    try {
        abs_path = std::filesystem::absolute(fmt::vformat(filename_format, arg_store)).string();
    } catch (const fmt::format_error& e) {
        abs_path.clear();
    }
    config.output_text(
        fmt::format("abs path: {}", abs_path.empty() ? "<invalid>" : abs_path.string()));
    config.output_text(fmt::format("variables: {}", fmt::join(variables, ", ")));

    config.st_separate("Single");
    record_next = config.config_bool("record_next");

    config.st_separate("Multiple");
    config.output_text(fmt::format(
        "current iteration: {}\ncurrent time: {}\nundersampling: {}\nestimated frametime: {:.2f}",
        record_enable ? fmt::to_string(iteration) : "stopped",
        record_enable ? fmt::format("{:.2f}", time_since_record.seconds()) : "stopped",
        undersampling, estimated_frametime_millis));
    const bool old_record_enable = record_enable;
    config.config_bool("enable", record_enable);
    if (record_enable && old_record_enable != record_enable)
        record();
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
        config.output_text("note: Iterations are 1-indexed");
    }
    if (trigger == 1) {
        config.config_float("framerate", record_framerate, "", 0.01);
        record_frametime_millis = 1000 / record_framerate;
        config.config_float("frametime", record_frametime_millis, "", 0.01);
        record_framerate = 1000 / record_frametime_millis;
    }
    config.st_separate();
    if (config.st_begin_child("advanced", "Advanced")) {
        config.config_uint("concurrency", max_concurrent_tasks, 1,
                           std::thread::hardware_concurrency(),
                           "Limit the maximum concurrency. Might be necessary with low memory.");
        config.config_percent("scale", scale);
        config.st_separate();
        config.config_int(
            "enable run", enable_run,
            "The specified run starts recording and resets the iteration and calls the "
            "configured callback and forces a rebuild if enabled.");

        config.config_bool("rebuild on record", rebuild_on_record,
                           "Rebuilds when recording starts");
        config.config_bool("callback after capture", callback_after_capture,
                           "calls the on_record callback after every capture");
        config.config_bool("callback on record", callback_on_record,
                           "calls the callback when the recording starts");
        config.st_separate();
        config.config_int("stop at run", stop_run,
                          "Stops recording at the specified run. -1 to disable.");
        config.config_int("stop at iteration", stop_iteration,
                          "Stops recording at the specified iteration. -1 to disable.");
        config.config_float(
            "stop after seconds", stop_after_seconds,
            "Stops recording after the specified seconds have passed. -1 to dissable.");
        config.config_int(
            "exit at run", exit_run,
            "Raises SIGTERM at the specified run. -1 to disable. Add a signal handler to "
            "shut down properly and not corrupt the images.");
        config.config_int("exit at iteration", exit_iteration,
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

void ImageWrite::set_callback(const std::function<void()> callback) {
    this->callback = callback;
}

} // namespace merian_nodes
