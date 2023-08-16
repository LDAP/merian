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

    if (record_next
        || (record_run_enable && record_run == (int)run.get_iteration())
        || (record_every_enable && frame % record_every == 0)) {

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

        run.add_submit_callback([this, image](const QueueHandle& queue) {
            queue->wait_idle();
            void* mem = image->get_memory()->map();

            std::string filename =
                fmt::format("{}_{:06}", this->base_filename, this->image_index++);

            std::filesystem::create_directories(std::filesystem::absolute(filename).parent_path());

            switch (this->format) {
            case FORMAT_PNG: {
                stbi_write_png(fmt::format("{}.png", filename).c_str(), image->get_extent().width,
                               image->get_extent().height, 4, mem, image->get_extent().width * 4);
                break;
            }
            case FORMAT_JPG: {
                stbi_write_jpg(fmt::format("{}.jpg", filename).c_str(), image->get_extent().width,
                               image->get_extent().height, 4, mem, 100);
                break;
            }
            case FORMAT_HDR: {
                stbi_write_hdr(fmt::format("{}.hdr", filename).c_str(), image->get_extent().width,
                               image->get_extent().height, 4, static_cast<float*>(mem));
                break;
            }
            }

            image->get_memory()->unmap();
        });

        if (force_rebuild)
            run.request_rebuild();
        record_next = false;
    }

    frame++;
}

void ImageWriteNode::get_configuration([[maybe_unused]] Configuration& config, bool&) {
    config.st_separate("General");
    config.config_options("format", format, {"PNG", "JPG", "HDR"},
                          Configuration::OptionsStyle::COMBO);
    config.config_bool("force rebuild", force_rebuild, "Forces a graph rebuild after every image");
    if (config.config_text("filename", buf.size(), buf.data())) {
        base_filename = buf.data();
    }
    config.output_text(fmt::format("abs path: {}", base_filename.empty() ? "<invalid>" : std::filesystem::absolute(base_filename).string()));

    config.st_separate("Single");
    record_next = config.config_bool("trigger");

    config.st_separate("Multiple");
    config.config_bool("record graph run", record_run_enable);
    config.st_no_space();
    config.config_int("iteration", record_run, "Save the result of the specified run.");
    record_run = std::max(record_run, 0);

    bool old_record_every_enable = record_every_enable;
    config.config_bool("record every", record_every_enable);
    config.st_no_space();
    config.config_int("frame", record_every, "Capture every i-th frame.");
    record_every = std::max(record_every, 1);
    if (old_record_every_enable != record_every_enable) {
        frame = 1;
    }
}

} // namespace merian
