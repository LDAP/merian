#pragma once

#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/graph/node.hpp"

#include <filesystem>

namespace merian {

class HDRImageRead : public Node {

  public:
    HDRImageRead();

    ~HDRImageRead();

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

    // can be nullptr when image is unloaded.
    float* image{nullptr};
    bool needs_run = true;

    int width, height, channels;
    std::filesystem::path filename;
    std::string config_filename;
};

} // namespace merian
