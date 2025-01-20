#pragma once

#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/connectors/managed_vk_image_out.hpp"
#include "merian-nodes/graph/node.hpp"

#include <optional>

namespace merian_nodes {

class AbstractABCompare : public Node {

  protected:
    AbstractABCompare(const std::optional<vk::Format> output_format = std::nullopt,
                      const std::optional<vk::Extent2D> output_extent = std::nullopt);

    virtual ~AbstractABCompare();

    std::vector<InputConnectorHandle> describe_inputs() override final;

  protected:
    const std::optional<vk::Format> output_format;
    const std::optional<vk::Extent2D> output_extent;

    const ManagedVkImageInHandle con_in_a = ManagedVkImageIn::transfer_src("a");
    const ManagedVkImageInHandle con_in_b = ManagedVkImageIn::transfer_src("b");
};

class ABSplit : public AbstractABCompare {

  public:
    ABSplit(const std::optional<vk::Format> output_format = std::nullopt,
            const std::optional<vk::Extent2D> output_extent = std::nullopt);

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    void process(GraphRun& run,
                 const CommandBufferHandle& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

  private:
    ManagedVkImageOutHandle con_out;
};

class ABSideBySide : public AbstractABCompare {

  public:
    ABSideBySide(const std::optional<vk::Format> output_format = std::nullopt,
                 const std::optional<vk::Extent2D> output_extent = std::nullopt);

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    void process(GraphRun& run,
                 const CommandBufferHandle& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

  private:
    ManagedVkImageOutHandle con_out;
};

} // namespace merian_nodes
