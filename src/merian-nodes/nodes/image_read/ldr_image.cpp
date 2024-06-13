#include "ldr_image.hpp"

#include "ext/stb_image.h"
#include <filesystem>

namespace merian_nodes {

LDRImageRead::LDRImageRead(const StagingMemoryManagerHandle& staging,
                           const std::filesystem::path& filename,
                           const bool linear,
                           const bool keep_on_host)
    : Node("LDR Image"), staging(staging), keep_on_host(keep_on_host), filename(filename) {

    assert(std::filesystem::exists(filename));
    if (!stbi_info(filename.string().c_str(), &width, &height, &channels)) {
        throw std::runtime_error{"format not supported!"};
    }

    format = linear ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;
}

LDRImageRead::~LDRImageRead() {
    if (image) {
        stbi_image_free(image);
    }
}

std::vector<OutputConnectorHandle>
LDRImageRead::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
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

            SPDLOG_INFO("Loaded image from {} ({}x{}, {} channels)", filename.string(), width, height,
                         channels);
        }

        staging->cmdToImage(cmd, *io[con_out], {0, 0, 0}, io[con_out]->get_extent(), first_layer(),
                            width * height * 4, image);

        if (!keep_on_host) {
            stbi_image_free(image);
            image = nullptr;
        }

        needs_run = false;
    }
}

LDRImageRead::NodeStatusFlags LDRImageRead::configuration(Configuration& config) {
    const vk::Format old_format = format;
    bool linear = format == vk::Format::eR8G8B8A8Unorm;
    config.config_bool("linear", linear);
    format = linear ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;
    config.config_bool("keep in host memory", keep_on_host, "");
    if (!keep_on_host && image) {
        stbi_image_free(image);
        image = nullptr;
    }

    const std::string text =
        fmt::format("filename: {}\nextent: {}x{}\nformat: {}\nhost cached: {}\n", filename.string(), width,
                    height, vk::to_string(format), image != nullptr);

    config.output_text(text);

    if (format != old_format) {
        return NEEDS_RECONNECT;
    } else {
        return {};
    }
}

} // namespace merian_nodes
