#include "merian-nodes/nodes/image_read/hdr_image.hpp"

#include "merian-nodes/graph/errors.hpp"
#include "stb_image.h"

#include <filesystem>

namespace merian {

HDRImageRead::HDRImageRead(const ContextHandle& context) : Node(), context(context) {}

HDRImageRead::~HDRImageRead() {
    if (image != nullptr) {
        stbi_image_free(image);
    }
}

std::vector<OutputConnectorHandle>
HDRImageRead::describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) {
    if (filename.empty()) {
        throw graph_errors::node_error{"no file set"};
    }
    if (!std::filesystem::exists(filename)) {
        throw graph_errors::node_error{fmt::format("file does not exist: {}", filename.string())};
    }
    if (stbi_info(filename.string().c_str(), &width, &height, &channels) == 0) {
        throw graph_errors::node_error{"format not supported!"};
    }

    con_out = ManagedVkImageOut::transfer_write("out", vk::Format::eR32G32B32A32Sfloat, width,
                                                height, 1, true);

    needs_run = true;
    return {con_out};
}

void HDRImageRead::process([[maybe_unused]] GraphRun& run,
                           [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    if (needs_run) {
        if (image == nullptr) {
            image = stbi_loadf(filename.string().c_str(), &width, &height, &channels, 4);
            assert(image);
            assert(width == (int)io[con_out]->get_extent().width &&
                   height == (int)io[con_out]->get_extent().height &&
                   1 == (int)io[con_out]->get_extent().depth);

            SPDLOG_INFO("Loaded image from {} ({}x{}, {} channels)", filename.string(), width,
                        height, channels);
        }

        run.get_allocator()->get_staging()->cmd_to_device(run.get_cmd(), io[con_out], image);

        if (!keep_on_host) {
            stbi_image_free(image);
            image = nullptr;
        }

        needs_run = false;
    }
}

HDRImageRead::NodeStatusFlags HDRImageRead::properties(Properties& config) {
    bool needs_rebuild = false;

    if (config.config_text("path", config_filename, true)) {
        needs_rebuild = true;
        filename = context->get_file_loader().find_file(config_filename).value_or(config_filename);
        if (image != nullptr) {
            stbi_image_free(image);
            image = nullptr;
        }
    }

    config.config_bool("keep in host memory", keep_on_host, "");
    if (!keep_on_host && (image != nullptr)) {
        stbi_image_free(image);
        image = nullptr;
    }

    const std::string text = fmt::format("filename: {}\nextent: {}x{}\nhost cached: {}\n",
                                         filename.string(), width, height, image != nullptr);

    config.output_text(text);

    if (needs_rebuild) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian
