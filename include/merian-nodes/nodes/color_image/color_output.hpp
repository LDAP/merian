#pragma once

#include "merian-nodes/connectors/managed_vk_image_out.hpp"
#include "merian-nodes/connectors/special_static_in.hpp"
#include "merian-nodes/graph/node.hpp"

namespace merian_nodes {

class ColorImage : public Node {

  public:
    ColorImage();

    ~ColorImage();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    vk::Format format = vk::Format::eR16G16B16A16Sfloat;
    bool extent_from_input = false;
    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};
    vk::ClearColorValue color = {};

    bool needs_run = true;
    ManagedVkImageOutHandle con_out;
    merian_nodes::SpecialStaticInHandle<vk::Extent3D> con_resolution =
        merian_nodes::SpecialStaticIn<vk::Extent3D>::create("resolution", true);
};

} // namespace merian_nodes
