#pragma once

#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/node.hpp"

namespace merian {

class ColorImage : public Node {

  public:
    ColorImage();

    ~ColorImage();

    std::vector<OutputConnectorDescriptor> describe_outputs(const NodeIOLayout& io_layout) override;

    [[nodiscard]] NodeStatusFlags
    process(const NodeIO& io, const NodeProcessInfo& info, Submission& submission) override;

    NodeStatusFlags properties(Properties& config) override;

  private:
    vk::Format format = vk::Format::eR16G16B16A16Sfloat;
    vk::Extent3D extent = vk::Extent3D{1920, 1080, 1};
    vk::ClearColorValue color = {};

    bool needs_run = true;
    ManagedVkImageOutHandle con_out;
};

} // namespace merian
