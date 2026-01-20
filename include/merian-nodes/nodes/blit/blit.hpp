#ifndef MERIAN_SHADERTOY_BLIT_HPP
#define MERIAN_SHADERTOY_BLIT_HPP
#include "merian-nodes/connectors/image/vk_image_in.hpp"
#include "merian-nodes/connectors/ptr_in.hpp"
#include "merian-nodes/graph/node.hpp"
#include "merian/vk/utils/blits.hpp"
#include "merian/vk/window/swapchain_manager.hpp"

namespace merian_nodes {

class Blit : public Node {
private:

public:
    Blit(const ContextHandle& context);

    virtual ~Blit() = default;

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    void
    process(GraphRun& run, const DescriptorSetHandle& descriptor_set, const NodeIO& io) override;

    NodeStatusFlags properties(Properties& config) override;

private:
    const ContextHandle context;

    uint32_t src_array_element = 0;
    uint32_t current_src_array_size = 1;

    BlitMode mode = FIT;

    VkImageInHandle src_image_in = VkImageIn::transfer_src("src", 0, true);
    //PtrInHandle<SwapchainAcquireResult> aquire_in = PtrIn<SwapchainAcquireResult>::create("aquire_in");
    VkImageInHandle dst_image_in = VkImageIn::transfer_dst("dst", 0, true);
};

} // namespace merian_nodes

#endif // MERIAN_SHADERTOY_BLIT_HPP
