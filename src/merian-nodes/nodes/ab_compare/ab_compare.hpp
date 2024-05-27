#pragma once

#include "merian-nodes/connectors/vk_image_in.hpp"
#include "merian-nodes/connectors/vk_image_out.hpp"
#include "merian-nodes/graph/node.hpp"

#include <optional>

namespace merian_nodes {

class ABCompareNode : public Node {

  protected:
    ABCompareNode(const std::string& name,
                  const std::optional<vk::Format> output_format = std::nullopt,
                  const std::optional<vk::Extent2D> output_extent = std::nullopt);

    virtual ~ABCompareNode();

    std::vector<InputConnectorHandle> describe_inputs() override final;

  protected:
    const std::optional<vk::Format> output_format;
    const std::optional<vk::Extent2D> output_extent;

    const VkImageInHandle img_in_a = VkImageIn::transfer_src("a");
    const VkImageInHandle img_in_b = VkImageIn::transfer_src("b");
};

class ABSplitNode : public ABCompareNode {

  public:
    ABSplitNode(const std::optional<vk::Format> output_format = std::nullopt,
                const std::optional<vk::Extent2D> output_extent = std::nullopt);

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

  private:
    VkImageOutHandle img_out;
};

class ABSideBySideNode : public ABCompareNode {

  public:
    ABSideBySideNode(const std::optional<vk::Format> output_format = std::nullopt,
                     const std::optional<vk::Extent2D> output_extent = std::nullopt);

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

  private:
    VkImageOutHandle img_out;
};

} // namespace merian_nodes
