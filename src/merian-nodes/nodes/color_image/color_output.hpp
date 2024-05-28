#pragma once

#include "merian-nodes/connectors/vk_image_out.hpp"
#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

class ColorImage : public Node {

  public:
    ColorImage(const vk::Format format,
                    const vk::Extent3D extent,
                    const vk::ClearColorValue color = {});

    ~ColorImage();

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

    NodeStatusFlags configuration(Configuration& config) override;

  private:
    const vk::ClearColorValue color;

    bool needs_run = true;
    VkImageOutHandle con_out;
};

} // namespace merian_nodes
