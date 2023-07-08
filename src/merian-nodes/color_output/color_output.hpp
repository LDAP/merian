#pragma once

#include "merian/vk/graph/node.hpp"

namespace merian {

class ColorOutputNode : public Node {

  public:
    ColorOutputNode(const vk::Format format,
                    const vk::Extent3D extent,
                    const vk::ClearColorValue color = {});

    ~ColorOutputNode();

    std::string name() override {
        return "Color Output";
    }

    std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                     const std::vector<NodeOutputDescriptorBuffer>&) override;

    void cmd_build(const vk::CommandBuffer& cmd,
                   const std::vector<std::vector<merian::ImageHandle>>&,
                   const std::vector<std::vector<merian::BufferHandle>>&,
                   const std::vector<std::vector<merian::ImageHandle>>& image_outputs,
                   const std::vector<std::vector<merian::BufferHandle>>&) override;

    void pre_process(NodeStatus& status) override;

    void get_configuration(Configuration& config) override;

  private:
    const vk::Format format;
    const vk::ClearColorValue color;
    const vk::Extent3D extent;
};

} // namespace merian
