#pragma once

#include "merian/vk/graph/node.hpp"
#include "merian/vk/utils/blits.hpp"

#include <optional>

namespace merian {

class ABCompareNode : public Node {

  protected:
    ABCompareNode(const std::optional<vk::Format> output_format = std::nullopt,
                  const std::optional<vk::Extent2D> output_extent = std::nullopt);

    virtual ~ABCompareNode();

    std::tuple<std::vector<NodeInputDescriptorImage>, std::vector<NodeInputDescriptorBuffer>>
    describe_inputs() override final;

  protected:
    const std::optional<vk::Format> output_format;
    const std::optional<vk::Extent2D> output_extent;
};

class ABSplitNode : public ABCompareNode {

  public:
    ABSplitNode(const std::optional<vk::Format> output_format = std::nullopt,
                const std::optional<vk::Extent2D> output_extent = std::nullopt);

    std::string name() override;

    std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                     const std::vector<NodeOutputDescriptorBuffer>&) override;

    void cmd_process(const vk::CommandBuffer& cmd,
                     GraphRun&,
                     const uint32_t,
                     const std::vector<ImageHandle>& image_inputs,
                     const std::vector<BufferHandle>&,
                     const std::vector<ImageHandle>& image_outputs,
                     const std::vector<BufferHandle>&) override;
};

class ABSideBySideNode : public ABCompareNode {

  public:
    ABSideBySideNode(const std::optional<vk::Format> output_format = std::nullopt,
                     const std::optional<vk::Extent2D> output_extent = std::nullopt);

    std::string name() override;

    std::tuple<std::vector<NodeOutputDescriptorImage>, std::vector<NodeOutputDescriptorBuffer>>
    describe_outputs(const std::vector<NodeOutputDescriptorImage>& connected_image_outputs,
                     const std::vector<NodeOutputDescriptorBuffer>&) override;

    void cmd_process(const vk::CommandBuffer& cmd,
                     GraphRun&,
                     const uint32_t,
                     const std::vector<ImageHandle>& image_inputs,
                     const std::vector<BufferHandle>&,
                     const std::vector<ImageHandle>& image_outputs,
                     const std::vector<BufferHandle>&) override;
};

} // namespace merian
