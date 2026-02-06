#pragma once

#include "merian-nodes/connectors/image/vk_image_in.hpp"
#include "merian-nodes/connectors/image/vk_image_out_managed.hpp"
#include "merian-nodes/graph/node.hpp"

#include <optional>

namespace merian {

class AbstractABCompare : public Node {

  protected:
    AbstractABCompare() = default;

    virtual ~AbstractABCompare();

    std::vector<InputConnectorHandle> describe_inputs() override final;

  protected:
    std::optional<vk::Format> output_format = std::nullopt;
    std::optional<vk::Extent2D> output_extent = std::nullopt;

    const VkImageInHandle con_in_a = VkImageIn::transfer_src("a");
    const VkImageInHandle con_in_b = VkImageIn::transfer_src("b");
};

class ABSplit : public AbstractABCompare {

  public:
    ABSplit() = default;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

  private:
    ManagedVkImageOutHandle con_out;
};

class ABSideBySide : public AbstractABCompare {

  public:
    ABSideBySide() = default;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

  private:
    ManagedVkImageOutHandle con_out;
};

} // namespace merian
