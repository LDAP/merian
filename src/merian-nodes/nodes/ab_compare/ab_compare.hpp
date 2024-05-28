#pragma once

#include "merian-nodes/connectors/vk_image_in.hpp"
#include "merian-nodes/connectors/vk_image_out.hpp"
#include "merian-nodes/graph/node.hpp"

#include <optional>

namespace merian_nodes {

class AbstractABCompare : public Node {

  protected:
    AbstractABCompare(const std::string& name,
                  const std::optional<vk::Format> output_format = std::nullopt,
                  const std::optional<vk::Extent2D> output_extent = std::nullopt);

    virtual ~AbstractABCompare();

    std::vector<InputConnectorHandle> describe_inputs() override final;

  protected:
    const std::optional<vk::Format> output_format;
    const std::optional<vk::Extent2D> output_extent;

    const VkImageInHandle con_in_a = VkImageIn::transfer_src("a");
    const VkImageInHandle con_in_b = VkImageIn::transfer_src("b");
};

class ABSplit : public AbstractABCompare {

  public:
    ABSplit(const std::optional<vk::Format> output_format = std::nullopt,
                const std::optional<vk::Extent2D> output_extent = std::nullopt);

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

  private:
    VkImageOutHandle con_out;
};

class ABSideBySide : public AbstractABCompare {

  public:
    ABSideBySide(const std::optional<vk::Format> output_format = std::nullopt,
                     const std::optional<vk::Extent2D> output_extent = std::nullopt);

    std::vector<OutputConnectorHandle>
    describe_outputs(const ConnectorIOMap& output_for_input) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

  private:
    VkImageOutHandle con_out;
};

} // namespace merian_nodes
