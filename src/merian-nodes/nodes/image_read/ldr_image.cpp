#include "ldr_image.hpp"

#include "ext/stb_image.h"
#include "merian-nodes/graph/errors.hpp"

#include <filesystem>

namespace merian_nodes {

LDRImageRead::LDRImageRead(const SharedContext& context) : Node(), context(context) {}

LDRImageRead::~LDRImageRead() {
    if (image) {
        stbi_image_free(image);
    }
}

std::vector<OutputConnectorHandle>
LDRImageRead::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
    if (filename.empty()) {
        throw graph_errors::node_error{"no file set"};
    }
    if (!std::filesystem::exists(filename)) {
        throw graph_errors::node_error{fmt::format("file does not exist: {}", filename.string())};
    }
    if (!stbi_info(filename.string().c_str(), &width, &height, &channels)) {
        throw graph_errors::node_error{"format not supported!"};
    }

    con_out = ManagedVkImageOut::transfer_write("out", format, width, height, 1, true);

    needs_run = true;
    return {con_out};
}

void LDRImageRead::process([[maybe_unused]] GraphRun& run,
                           const vk::CommandBuffer& cmd,
                           [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    if (needs_run) {
        if (!image) {
            image = stbi_load(filename.string().c_str(), &width, &height, &channels, 4);
            assert(image);
            assert(width == (int)io[con_out]->get_extent().width &&
                   height == (int)io[con_out]->get_extent().height);

            SPDLOG_INFO("Loaded image from {} ({}x{}, {} channels)", filename.string(), width,
                        height, channels);
        }

        run.get_allocator()->getStaging()->cmdToImage(cmd, *io[con_out], {0, 0, 0},
                                                      io[con_out]->get_extent(), first_layer(),
                                                      width * height * 4, image);

        if (!keep_on_host) {
            stbi_image_free(image);
            image = nullptr;
        }

        needs_run = false;
    }
}

LDRImageRead::NodeStatusFlags LDRImageRead::properties(Properties& config) {
    bool needs_rebuild = false;

    if (config.config_text("path", config_filename.size(), config_filename.data(), true)) {
        needs_rebuild = true;
        filename =
            context->loader.find_file(config_filename.data()).value_or(config_filename.data());
        if (image) {
            stbi_image_free(image);
            image = nullptr;
        }
    }

    const vk::Format old_format = format;
    bool linear = format == vk::Format::eR8G8B8A8Unorm;
    config.config_bool("linear", linear);
    format = linear ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;
    needs_rebuild |= format != old_format;
    config.config_bool("keep in host memory", keep_on_host, "");
    if (!keep_on_host && image) {
        stbi_image_free(image);
        image = nullptr;
    }

    const std::string text =
        fmt::format("filename: {}\nextent: {}x{}\nformat: {}\nhost cached: {}\n", filename.string(),
                    width, height, vk::to_string(format), image != nullptr);

    config.output_text(text);

    if (needs_rebuild) {
        return NEEDS_RECONNECT;
    }
    return {};
}

} // namespace merian_nodes
