#pragma once

#include "merian/vk/graph/node.hpp"

namespace merian {

// An accumulate node for vk::Format::eR32G32B32A32Sfloat images
class AccumulateF32ImageNode : public Node {

    std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override;

    virtual std::tuple<std::vector<merian::NodeOutputDescriptorImage>,
                       std::vector<merian::NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<merian::NodeOutputDescriptorImage>& connected_image_outputs,
                     const std::vector<merian::NodeOutputDescriptorBuffer>&) override;

};

} // namespace merian
