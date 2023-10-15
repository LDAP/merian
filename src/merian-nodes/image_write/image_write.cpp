#include "image_write.hpp"
#include "ext/stb_image_write.h"
#include "merian/vk/graph/graph.hpp"
#include "merian/vk/utils/blits.hpp"
#include <filesystem>

namespace merian {

#define FORMAT_PNG 0
#define FORMAT_JPG 1
#define FORMAT_HDR 2

ImageWriteNode::ImageWriteNode(const SharedContext context,
                               const ResourceAllocatorHandle allocator,
                               const std::string& base_filename = "image")
    : context(context), allocator(allocator), base_filename(base_filename), buf(256) {
    assert(base_filename.size() < buf.size());
    std::copy(base_filename.begin(), base_filename.end(), buf.begin());
}

ImageWriteNode::~ImageWriteNode() {}

std::string ImageWriteNode::name() {
    return "Image Write";
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

void ImageWriteNode::pre_process([[maybe_unused]] const uint64_t& iteration,
                                 [[maybe_unused]] NodeStatus& status) {
    if (!record_enable && ((int64_t)iteration == trigger_run)) {
        record_enable = true;
        status.request_rebuild |= rebuild_on_record;
        this->iteration = 0;
        if (callback_on_record && callback)
            callback();
    }
};

void ImageWriteNode::cmd_process([[maybe_unused]] const vk::CommandBuffer& cmd,
                                 [[maybe_unused]] GraphRun& run,
                                 [[maybe_unused]] const uint32_t set_index,
                                 [[maybe_unused]] const std::vector<ImageHandle>& image_inputs,
                                 [[maybe_unused]] const std::vector<BufferHandle>& buffer_inputs,
                                 [[maybe_unused]] const std::vector<ImageHandle>& image_outputs,
                                 [[maybe_unused]] const std::vector<BufferHandle>& buffer_outputs) {
    if (base_filename.empty()) {
        return;
    }

    if (record_next || (record_enable && record_iteration == iteration)) {

        vk::Format format = this->format == FORMAT_HDR ? vk::Format::eR32G32B32A32Sfloat
                                                       : vk::Format::eR8G8B8A8Srgb;
        vk::ImageCreateInfo info{
            {},
            vk::ImageType::e2D,
            format,
            image_inputs[0]->get_extent(),
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

        ImageHandle image = allocator->createImage(info, MemoryMappingType::HOST_ACCESS_RANDOM);

        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
                            image->barrier(vk::ImageLayout::eTransferDstOptimal, {},
                                           vk::AccessFlagBits::eTransferWrite));

        cmd_blit_stretch(cmd, *image_inputs[0], image_inputs[0]->get_current_layout(),
                         image_inputs[0]->get_extent(), *image,
                         vk::ImageLayout::eTransferDstOptimal, image_inputs[0]->get_extent());

        cmd.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, {}, {},
            image->barrier(vk::ImageLayout::eGeneral, vk::AccessFlagBits::eTransferWrite,
                           vk::AccessFlagBits::eHostRead));

        int it = iteration;
        int run_it = run.get_iteration();
        int image_index = this->image_index++;
        run.add_submit_callback([this, image, it, image_index, run_it](const QueueHandle& queue) {
            queue->wait_idle();
            void* mem = image->get_memory()->map();

            std::string filename =
                fmt::format("{}_{:06}_{:06}_{:06}", this->base_filename, it, image_index, run_it);
            char* tmp_filename = std::tmpnam(NULL);

            switch (this->format) {
            case FORMAT_PNG: {
                filename += ".png";
                stbi_write_png(tmp_filename, image->get_extent().width, image->get_extent().height,
                               4, mem, image->get_extent().width * 4);
                break;
            }
            case FORMAT_JPG: {
                filename += ".jpg";
                stbi_write_jpg(tmp_filename, image->get_extent().width, image->get_extent().height,
                               4, mem, 100);
                break;
            }
            case FORMAT_HDR: {
                filename += ".hdr";
                stbi_write_hdr(tmp_filename, image->get_extent().width, image->get_extent().height,
                               4, static_cast<float*>(mem));
                break;
            }
            }

            std::filesystem::create_directories(std::filesystem::absolute(filename).parent_path());
            std::filesystem::rename(tmp_filename, filename);

            image->get_memory()->unmap();
        });

        if (rebuild_after_capture)
            run.request_rebuild();
        if (callback_after_capture && callback)
            callback();
        record_next = false;

        record_iteration *= record_enable ? it_power : 1;
        record_iteration += record_enable ? it_offset : 0;
    }

    iteration++;
}

void ImageWriteNode::get_configuration([[maybe_unused]] Configuration& config,
                                       bool& needs_rebuild) {
    config.st_separate("General");
    config.config_options("format", format, {"PNG", "JPG", "HDR"},
                          Configuration::OptionsStyle::COMBO);
    config.config_bool("rebuild after capture", rebuild_after_capture,
                       "forces a graph rebuild after every capture");
    config.config_bool("rebuild on record", rebuild_on_record, "Rebuilds when recording starts");
    config.config_bool("callback after capture", callback_after_capture,
                       "calls the on_record callback after every capture");
    config.config_bool("callback on record", callback_on_record,
                       "calls the callback when the recording starts");
    if (config.config_text("filename", buf.size(), buf.data())) {
        base_filename = buf.data();
    }
    config.output_text(fmt::format(
        "abs path: {}",
        base_filename.empty() ? "<invalid>" : std::filesystem::absolute(base_filename).string()));

    config.st_separate("Single");
    record_next = config.config_bool("trigger");

    config.st_separate("Multiple");
    config.output_text(fmt::format("current iteration: {}",
                                   record_enable ? fmt::to_string(iteration) : "stopped"));

    bool old_record_enable = record_enable;
    config.config_bool("enable", record_enable);
    if (record_enable && !old_record_enable) {
        needs_rebuild |= rebuild_on_record;
        iteration = 0;
        if (callback)
            callback();
    }
    config.config_int("run trigger", trigger_run,
                      "The specified run starts recording and resets the iteration and calls the "
                      "configured callback and forces a rebuild if enabled.");

    config.st_separate();

    config.config_int("iteration", record_iteration,
                      "Save the result of of the the specified iteration");
    record_iteration = std::max(record_iteration, 0);

    config.config_int("iteration power", it_power,
                      "Multiplies the iteration specifier with this value after every capture");
    config.config_int("iteration offset", it_offset,
                      "Adds this value to the iteration specifier after every capture. (After "
                      "applying the power).");
}

void ImageWriteNode::set_callback(const std::function<void()> callback) {
    this->callback = callback;
}

} // namespace merian
