#pragma once

#include "merian-nodes/graph/node.hpp"
#include "merian-nodes/connectors/vk_image_out.hpp"

#include <filesystem>

namespace merian_nodes {

class LDRImageRead : public Node {

  public:
    // By default images are interpretet as sRGB, turn on linear if you want to load images for
    // normal maps, depth, and such.
    //
    // Set keep_on_host to keep a copy in host memory, otherwise the image is reloaded from disk
    // everytime the graph reconnects.
    LDRImageRead(const StagingMemoryManagerHandle& staging,
              const std::filesystem::path& path,
              const bool linear = false,
              const bool keep_on_host = false);

    ~LDRImageRead();

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

    NodeStatusFlags configuration(Configuration& config) override;

  private:
    const StagingMemoryManagerHandle staging;
    bool keep_on_host;

    VkImageOutHandle con_out;

    vk::Format format;
    // can be nullptr when image is unloaded.
    unsigned char* image{nullptr};
    bool needs_run = true;

    int width, height, channels;
    std::filesystem::path filename;
};

} // namespace merian_nodes
