#pragma once

#include "merian-nodes/graph/node.hpp"

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

    void cmd_build(const vk::CommandBuffer& cmd, const std::vector<NodeIO>& ios) override;

    void cmd_process(const vk::CommandBuffer& cmd,
                     GraphRun& run,
                     const std::shared_ptr<FrameData>& frame_data,
                     const uint32_t set_index,
                     const NodeIO& io) override;

    void pre_process([[maybe_unused]] const uint64_t& iteration, NodeStatus& status) override;

    void get_configuration(Configuration& config, bool& needs_rebuild) override;

  private:
    const vk::Format format;
    const vk::ClearColorValue color;
    const vk::Extent3D extent;
    bool needs_run = true;
};

} // namespace merian