#include "merian-nodes/nodes/image_read/hdr_image.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "merian/io/image_io.hpp"

#include <filesystem>

namespace merian {

HDRImageRead::HDRImageRead() {}

void HDRImageRead::initialize(const ContextHandle& context,
                              const ResourceAllocatorHandle& /*allocator*/) {
    this->context = context;
}

HDRImageRead::~HDRImageRead() = default;

std::vector<OutputConnectorDescriptor>
HDRImageRead::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    if (filename.empty()) {
        throw graph_errors::node_error{"no file set"};
    }
    if (!std::filesystem::exists(filename)) {
        throw graph_errors::node_error{fmt::format("file does not exist: {}", filename.string())};
    }

    try {
        ImageInfo info;
        image = image_load_f32(filename, info, 4);
        width = info.width;
        height = info.height;
        channels = info.channels;
    } catch (const std::runtime_error& e) {
        throw graph_errors::node_error{e.what()};
    }

    con_out =
        ManagedVkImageOut::transfer_write(vk::Format::eR32G32B32A32Sfloat, width, height, 1, true);
    needs_run = true;
    return {{"out", con_out}};
}

void HDRImageRead::process([[maybe_unused]] GraphRun& run,
                           [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    if (!needs_run) {
        return;
    }
    if (!image) {
        ImageInfo info;
        image = image_load_f32(filename, info, 4);
    }
    SPDLOG_INFO("Loaded image from {} ({}x{}, {} channels)", filename.string(), width, height,
                channels);

    run.get_allocator()->get_staging()->cmd_to_device(run.get_cmd(), io[con_out], image->get_data());

    if (!keep_on_host) {
        image.reset();
    }
    needs_run = false;
}

HDRImageRead::NodeStatusFlags HDRImageRead::properties(Properties& config) {
    bool needs_rebuild = false;

    if (config.config_text("path", config_filename, true)) {
        needs_rebuild = true;
        filename = context->get_file_loader()->find_file(config_filename).value_or(config_filename);
        image.reset();
    }

    config.config_bool("keep in host memory", keep_on_host, "");
    if (!keep_on_host) {
        image.reset();
    }

    config.output_text(fmt::format("filename: {}\nextent: {}x{}\nhost cached: {}\n",
                                   filename.string(), width, height, image != nullptr));

    if (needs_rebuild) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
