#pragma once

#include "merian-nodes/graph/connector_output.hpp"
#include "merian-nodes/resources/image_array_resource.hpp"

namespace merian_nodes {

class VkImageOut;
using VkImageOutHandle = std::shared_ptr<VkImageOut>;

// Output a Vulkan image that is allocated and managed by the graph.
// Note that it only supplies a descriptor if stage_flags contains at least one bit.
class VkImageOut : public TypedOutputConnector<ImageArrayResource&> {
  public:
    VkImageOut(const std::string& name,
                      const bool persistent = false,
                      const uint32_t array_size = 1);

    virtual ImageArrayResource& resource(const GraphResourceHandle& resource) override;

    uint32_t array_size() const;
  public:
    std::vector<merian::ImageHandle> images;
    const bool persistent;
};

} // namespace merian_nodes
