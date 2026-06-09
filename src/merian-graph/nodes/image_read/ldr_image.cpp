#include "merian-graph/nodes/image_read/ldr_image.hpp"

#include "merian-graph/graph/errors.hpp"
#include "merian/io/image_io.hpp"

#include <filesystem>

namespace merian {

LDRImageRead::LDRImageRead() {}

void LDRImageRead::initialize(const ContextHandle& context,
                              const ResourceAllocatorHandle& /*allocator*/) {
    this->context = context;
}

LDRImageRead::~LDRImageRead() = default;

std::vector<OutputConnectorDescriptor>
LDRImageRead::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    if (filename.empty()) {
        throw graph_errors::node_error{"no file set"};
    }
    if (!std::filesystem::exists(filename)) {
        throw graph_errors::node_error{fmt::format("file does not exist: {}", filename.string())};
    }

    try {
        ImageInfo info;
        image = image_load_u8(filename, info, 4);
        width = info.width;
        height = info.height;
        channels = info.channels;
    } catch (const std::runtime_error& e) {
        throw graph_errors::node_error{e.what()};
    }

    con_out = ManagedVkImageOut::transfer_write(format, width, height, 1, true);
    needs_run = true;
    return {{"out", con_out}};
}

void LDRImageRead::process([[maybe_unused]] GraphRun& run,
                           [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    if (!needs_run) {
        return;
    }
    if (!image) {
        ImageInfo info;
        image = image_load_u8(filename, info, 4);
    }
    SPDLOG_INFO("Loaded image from {} ({}x{}, {} channels)", filename.string(), width, height,
                channels);

    run.get_allocator()->get_staging()->cmd_to_device(run.get_cmd(), io[con_out], image->get_data());

    if (!keep_on_host) {
        image.reset();
    }
    needs_run = false;
}

LDRImageRead::NodeStatusFlags LDRImageRead::properties(Properties& config) {
    bool needs_rebuild = false;

    if (config.config_text("path", config_filename, true)) {
        needs_rebuild = true;
        filename = context->get_file_loader()->find_file(config_filename).value_or(config_filename);
        image.reset();
    }

    const vk::Format old_format = format;
    bool linear = format == vk::Format::eR8G8B8A8Unorm;
    config.config_bool("linear", linear);
    format = linear ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;
    needs_rebuild |= format != old_format;
    config.config_bool("keep in host memory", keep_on_host, "");
    if (!keep_on_host) {
        image.reset();
    }

    config.output_text(fmt::format("filename: {}\nextent: {}x{}\nformat: {}\nhost cached: {}\n",
                                   filename.string(), width, height, vk::to_string(format),
                                   image != nullptr));

    if (needs_rebuild) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
