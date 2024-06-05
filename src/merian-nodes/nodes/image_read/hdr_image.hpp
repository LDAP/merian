#pragma once

#include "merian-nodes/connectors/vk_image_out.hpp"
#include "merian-nodes/graph/node.hpp"

#include <filesystem>

namespace merian_nodes {

class HDRImageRead : public Node {

  public:
    // By default images are interpretet as sRGB, turn on linear if you want to load images for
    // normal maps, depth, and such.
    //
    // Set keep_on_host to keep a copy in host memory, otherwise the image is reloaded from disk
    // everytime the graph reconnects.
    HDRImageRead(const StagingMemoryManagerHandle& staging,
                 const std::filesystem::path& path,
                 const bool keep_on_host = false);

    ~HDRImageRead();

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

    // can be nullptr when image is unloaded.
    float* image{nullptr};
    bool needs_run = true;

    int width, height, channels;
    std::filesystem::path filename;
};

} // namespace merian_nodes
