#pragma once

#include "merian-nodes/connectors/managed_vk_image_out.hpp"
#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

class ColorImage : public Node {

  public:
    ColorImage(const vk::Format format = vk::Format::eR16G16B16A16Sfloat,
               const vk::Extent3D extent = vk::Extent3D{1920, 1080, 1},
               const vk::ClearColorValue color = {});

    ~ColorImage();

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    const vk::ClearColorValue color;

    bool needs_run = true;
    ManagedVkImageOutHandle con_out;
};

} // namespace merian_nodes
