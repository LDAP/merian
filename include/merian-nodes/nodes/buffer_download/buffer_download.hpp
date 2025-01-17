#pragma once

#include "merian-nodes/connectors/managed_vk_buffer_in.hpp"
#include "merian-nodes/connectors/managed_vk_buffer_out.hpp"
#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/graph/node.hpp"

#include <merian-nodes/connectors/ptr_out.hpp>
#include <merian-nodes/nodes/ab_compare/ab_compare.hpp>

namespace merian_nodes {

class BufferDownload : public Node {
private:
    static constexpr uint32_t local_size_x = 16;
    static constexpr uint32_t local_size_y = 16;
    static constexpr uint32_t workgroup_size = local_size_x * local_size_y;

    struct PushConstant {
        uint32_t divisor;

        int size;
        int offset;
        int count;
    };

public:
    BufferDownload(const ContextHandle context);

    ~BufferDownload();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 const DescriptorSetLayoutHandle& descriptor_set_layout) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

private:
    const ContextHandle context;

    ManagedVkBufferInHandle con_src = ManagedVkBufferIn::transfer_src("src");
    PtrOutHandle<const void*> con_out = PtrOut<const void*>::create("out");

    std::vector<const void*> results;
};

} // namespace merian_nodes
