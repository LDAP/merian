#include "hdr_image.hpp"

#include "ext/stb_image.h"
#include <filesystem>

namespace merian_nodes {

HDRImageRead::HDRImageRead(const StagingMemoryManagerHandle& staging,
                           const std::filesystem::path& filename,
                           const bool keep_on_host)
    : Node("HDR Image"), staging(staging), keep_on_host(keep_on_host), filename(filename) {

    assert(std::filesystem::exists(filename));
    if (!stbi_info(filename.string().c_str(), &width, &height, &channels)) {
        throw std::runtime_error{"format not supported!"};
    }

    con_out =
        VkImageOut::transfer_write("out", vk::Format::eR32G32B32A32Sfloat, width, height, 1, true);
}

HDRImageRead::~HDRImageRead() {
    if (image) {
        stbi_image_free(image);
    }
}

std::vector<OutputConnectorHandle>
HDRImageRead::describe_outputs([[maybe_unused]] const ConnectorIOMap& output_for_input) {
    needs_run = true;
    return {con_out};
}

void HDRImageRead::process([[maybe_unused]] GraphRun& run,
                           const vk::CommandBuffer& cmd,
                           [[maybe_unused]] const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    if (needs_run) {
        if (!image) {
            image = stbi_loadf(filename.string().c_str(), &width, &height, &channels, 4);
            assert(image);
            assert(width == (int)io[con_out]->get_extent().width &&
                   height == (int)io[con_out]->get_extent().height);

            SPDLOG_INFO("Loaded image from {} ({}x{}, {} channels)", filename.string(), width, height,
                         channels);
        }

        staging->cmdToImage(cmd, *io[con_out], {0, 0, 0}, io[con_out]->get_extent(), first_layer(),
                            width * height * 4 * sizeof(float), image);

        if (!keep_on_host) {
            stbi_image_free(image);
            image = nullptr;
        }

        needs_run = false;
    }
}

HDRImageRead::NodeStatusFlags HDRImageRead::configuration(Configuration& config) {
    config.config_bool("keep in host memory", keep_on_host, "");
    if (!keep_on_host && image) {
        stbi_image_free(image);
        image = nullptr;
    }

    const std::string text = fmt::format("filename: {}\nextent: {}x{}\nhost cached: {}\n", filename.string(),
                                         width, height, image != nullptr);

    config.output_text(text);

    return {};
}

} // namespace merian_nodes
