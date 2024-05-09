#pragma once

#include "merian/io/file_loader.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/vk/memory/resource_allocator.hpp"

namespace merian {

class ImageNode : public Node {

  public:
    // turn on linear if you want to load images for normal maps, depth,....
    // else the images are interpreted as SRGB.
    ImageNode(const ResourceAllocatorHandle allocator,
              const std::string path,
              const FileLoader loader,
              const bool linear = false);

    ~ImageNode();

    std::string name() override {
        return "Image";
    }

    std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(
        const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
        const std::vector<NodeOutputDescriptorBuffer>& connected_buffer_outputs) override;

    void pre_process([[maybe_unused]] const uint64_t& iteration, NodeStatus& status) override;

    void cmd_build(const vk::CommandBuffer& cmd, const std::vector<NodeIO>& ios) override;

    void get_configuration(Configuration& config, bool& needs_rebuild) override;

  private:
    const ResourceAllocatorHandle allocator;

    vk::Format format;
    unsigned char* image;
    int width, height, channels;
    std::string filename;
};

} // namespace merian
