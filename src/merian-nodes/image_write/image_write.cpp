#include "image_write.hpp"
#include "ext/stb_image_write.h"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/utils/blits.hpp"
#include <csignal>
#include <filesystem>

namespace merian {

#define FORMAT_PNG 0
#define FORMAT_JPG 1
#define FORMAT_HDR 2

ImageWriteNode::ImageWriteNode(const SharedContext context,
                               const ResourceAllocatorHandle allocator,
                               const std::string& filename_format)
    : context(context), allocator(allocator), filename_format(filename_format), buf(1024) {
    assert(filename_format.size() < buf.size());
    std::copy(filename_format.begin(), filename_format.end(), buf.begin());
}

ImageWriteNode::~ImageWriteNode() {}

std::string ImageWriteNode::name() {
    return "Image Write";
}

std::shared_ptr<Node::FrameData> ImageWriteNode::create_frame_data() {
    return std::make_shared<FrameData>();
}

// Declare the inputs that you require
std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
ImageWriteNode::describe_inputs() {
    return {
        {
            NodeInputDescriptorImage::transfer_src("src"),
        },
        {},
    };
}

void ImageWriteNode::record() {
    record_enable = true;
    needs_rebuild |= rebuild_on_record;
    this->iteration = 1;
    if (callback_on_record && callback)
        callback();
}

void ImageWriteNode::pre_process([[maybe_unused]] const uint64_t& run_iteration,
                                 [[maybe_unused]] NodeStatus& status) {
    if (!record_enable && ((int64_t)run_iteration == trigger_run)) {
        record();
    }
    status.request_rebuild = needs_rebuild;
    needs_rebuild = false;
};

void ImageWriteNode::cmd_process(
    const vk::CommandBuffer& cmd,
    GraphRun& run,
    [[maybe_unused]] const std::shared_ptr<Node::FrameData>& node_frame_data,
    [[maybe_unused]] const uint32_t set_index,
    const NodeIO& io) {
    if (filename_format.empty()) {
        record_enable = false;
        record_next = false;
    }

    if (record_next || (record_enable && record_iteration == iteration)) {
        const vk::Format format = this->format == FORMAT_HDR ? vk::Format::eR32G32B32A32Sfloat
                                                             : vk::Format::eR8G8B8A8Srgb;
        const vk::FormatProperties format_properties =
            context->physical_device.physical_device.getFormatProperties(format);
        const std::shared_ptr<FrameData> frame_data =
            std::static_pointer_cast<FrameData>(node_frame_data);

        const ImageHandle src = io.image_inputs[0];
        vk::Extent3D scaled = max(multiply(src->get_extent(), scale), {1, 1, 1});

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
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
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
                frame_data->intermediate_image = intermediate_image;

                cmd.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {},
                    {}, {},
                    intermediate_image->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                                                vk::AccessFlagBits::eTransferWrite));
                cmd_blit_stretch(cmd, *src, src->get_current_layout(), src->get_extent(),
                                 *intermediate_image, vk::ImageLayout::eTransferDstOptimal,
                                 intermediate_image->get_extent());
                cmd.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {},
                    {}, {},
                    intermediate_image->barrier(vk::ImageLayout::eTransferSrcOptimal,
                                                vk::AccessFlagBits::eTransferWrite,
                                                vk::AccessFlagBits::eTransferRead));
            }
            {
                MERIAN_PROFILE_SCOPE_GPU(run.get_profiler(), cmd, "copy to linear image");
                cmd.copyImage(*frame_data->intermediate_image.value(),
                              frame_data->intermediate_image.value()->get_current_layout(),
                              *linear_image, linear_image->get_current_layout(),
                              vk::ImageCopy(first_layer(), {}, first_layer(), {},
                                            frame_data->intermediate_image.value()->get_extent()));
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

        int it = iteration;
        int run_it = run.get_iteration();
        int image_index = this->image_index++;
        std::string filename_format = this->filename_format;
        const std::function<void()> write_task =
            ([this, image_ready, linear_image, it, image_index, run_it, filename_format]() {
                image_ready->wait(1);
                void* mem = linear_image->get_memory()->map();

                std::filesystem::path path = std::filesystem::absolute(fmt::format(
                    fmt::runtime(filename_format), fmt::arg("record_iteration", it),
                    fmt::arg("image_index", image_index), fmt::arg("run_iteration", run_it),
                    fmt::arg("width", linear_image->get_extent().width),
                    fmt::arg("height", linear_image->get_extent().height)));
                std::filesystem::create_directories(path.parent_path());
                const std::string tmp_filename =
                    (path.parent_path() / (".interm_" + path.filename().string())).string();

                switch (this->format) {
                case FORMAT_PNG: {
                    path += ".png";
                    stbi_write_png(tmp_filename.c_str(), linear_image->get_extent().width,
                                   linear_image->get_extent().height, 4, mem,
                                   linear_image->get_extent().width * 4);
                    break;
                }
                case FORMAT_JPG: {
                    path += ".jpg";
                    stbi_write_jpg(tmp_filename.c_str(), linear_image->get_extent().width,
                                   linear_image->get_extent().height, 4, mem, 100);
                    break;
                }
                case FORMAT_HDR: {
                    path += ".hdr";
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
            run.request_rebuild();
        if (callback_after_capture && callback)
            callback();
        record_next = false;

        record_iteration *= record_enable ? it_power : 1;
        record_iteration += record_enable ? it_offset : 0;
    }

    if (stop_run == (int64_t)run.get_iteration() || stop_iteration == iteration) {
        record_enable = false;
    }
    if (exit_run == (int64_t)run.get_iteration() || exit_iteration == iteration) {
        raise(SIGKILL);
    }

    iteration++;
}

void ImageWriteNode::get_configuration([[maybe_unused]] Configuration& config, bool&) {
    config.st_separate("General");
    config.config_options("format", format, {"PNG", "JPG", "HDR"},
                          Configuration::OptionsStyle::COMBO);
    config.config_uint("concurrency", max_concurrent_tasks, 1, std::thread::hardware_concurrency(),
                       "Limit the maximum concurrency. Might be necessary with low memory.");
    config.config_percent("scale", scale);
    config.config_bool("rebuild after capture", rebuild_after_capture,
                       "forces a graph rebuild after every capture");
    config.config_bool("rebuild on record", rebuild_on_record, "Rebuilds when recording starts");
    config.config_bool("callback after capture", callback_after_capture,
                       "calls the on_record callback after every capture");
    config.config_bool("callback on record", callback_on_record,
                       "calls the callback when the recording starts");
    if (config.config_text("filename", buf.size(), buf.data(), false,
                           "Provide a format string for the path. Supported variables are: "
                           "record_iteration, run_iteration, image_index, width, height")) {
        filename_format = buf.data();
    }
    config.output_text(
        fmt::format("abs path: {}", filename_format.empty()
                                        ? "<invalid>"
                                        : std::filesystem::absolute(filename_format).string()));

    config.st_separate("Single");
    record_next = config.config_bool("trigger");

    config.st_separate("Multiple");
    config.output_text(fmt::format("current iteration: {}",
                                   record_enable ? fmt::to_string(iteration) : "stopped"));
    const bool old_record_enable = record_enable;
    config.config_bool("enable", record_enable);
    if (record_enable && old_record_enable != record_enable)
        record();
    config.config_int("run trigger", trigger_run,
                      "The specified run starts recording and resets the iteration and calls the "
                      "configured callback and forces a rebuild if enabled.");

    config.st_separate();

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
    config.st_separate();
    config.config_int("stop at run", stop_run,
                      "Stops recording at the specified run. -1 to disable.");
    config.config_int("stop at iteration", stop_iteration,
                      "Stops recording at the specified iteration. -1 to disable.");
    config.config_int("exit at run", exit_run,
                      "Raises SIGKILL at the specified run. -1 to disable. Add a signal handler to "
                      "shut down properly and not corrupt the images.");
    config.config_int("exit at iteration", exit_iteration,
                      "Raises SIGKILL at the specified iteration. -1 to disable. Add a signal "
                      "handler to shut down properly and not corrupt the images.");
}

void ImageWriteNode::set_callback(const std::function<void()> callback) {
    this->callback = callback;
}

} // namespace merian
