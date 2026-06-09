#pragma once

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/node.hpp"
#include "merian/utils/blob.hpp"

#include <filesystem>

namespace merian {

class LDRImageRead : public Node {

  public:
    LDRImageRead();

    ~LDRImageRead() override;

    void initialize(const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override;

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    ContextHandle context;
    bool keep_on_host = false;

    ManagedVkImageOutHandle con_out;

    vk::Format format = vk::Format::eR8G8B8A8Srgb;
    // null when image is unloaded.
    BlobHandle image;
    bool needs_run = true;

    int width = 0;
    int height = 0;
    int channels = 0;
    std::filesystem::path filename;
    std::string config_filename;
};

} // namespace merian
