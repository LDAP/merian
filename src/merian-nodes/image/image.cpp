#include "image.hpp"

#include "ext/stb_image.h"

namespace merian {

ImageNode::ImageNode(const ResourceAllocatorHandle allocator,
                     const std::string path,
                     const FileLoader loader,
                     const bool linear)
    : allocator(allocator) {

    auto file = loader.find_file(path);
    assert(file.has_value());
    filename = file.value().string();

    image = stbi_load(file->string().c_str(), &width, &height, &channels, 4);
    assert(image);
    SPDLOG_DEBUG("Loaded image from {} ({}x{}, {} channels)", file->string(), width, height,
                 channels);

    format = linear ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR8G8B8A8Srgb;
}

ImageNode::~ImageNode() {
    stbi_image_free(image);
}

std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
ImageNode::describe_outputs(const std::vector<NodeOutputDescriptorImage>&,
                            const std::vector<NodeOutputDescriptorBuffer>&) {
    return {{NodeOutputDescriptorImage::transfer_write("output", format, width, height, true)}, {}};
}

void ImageNode::pre_process([[maybe_unused]] const uint64_t& iteration, NodeStatus& status) {
    status.skip_run = true;
}

void ImageNode::cmd_build(const vk::CommandBuffer& cmd, const std::vector<NodeIO>& ios) {
    allocator->getStaging()->cmdToImage(cmd, *ios[0].image_outputs[0], {0, 0, 0},
                                        ios[0].image_outputs[0]->get_extent(), first_layer(),
                                        width * height * 4, image);
}

void ImageNode::get_configuration(Configuration& config, bool&) {
    std::string text;
    text += fmt::format("filename: {}\n", filename);
    text += fmt::format("extent: {}x{}\n", width, height);
    text += fmt::format("format: {}\n", vk::to_string(format));

    config.output_text(text);
}

} // namespace merian
