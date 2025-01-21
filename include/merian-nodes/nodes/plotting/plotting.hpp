#pragma once

#include "merian-nodes/connectors/managed_vk_buffer_out.hpp"
#include "merian-nodes/connectors/managed_vk_image_in.hpp"
#include "merian-nodes/graph/node.hpp"

#include "merian/vk/memory/resource_allocator.hpp"
#include "merian/vk/pipeline/pipeline.hpp"
#include "merian/vk/shader/shader_module.hpp"
#include "merian-nodes/nodes/plotting/plotting_types.hpp"

#include <merian-nodes/connectors/ptr_in.hpp>

namespace merian_nodes {

class Plotting : public Node {
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
    Plotting(const ContextHandle context);

    ~Plotting();

    std::vector<InputConnectorHandle> describe_inputs() override;

    std::vector<OutputConnectorHandle> describe_outputs(const NodeIOLayout& io_layout) override;

    NodeStatusFlags properties(Properties& config) override;

    void process(GraphRun& run,
                 const vk::CommandBuffer& cmd,
                 const DescriptorSetHandle& descriptor_set,
                 const NodeIO& io) override;

private:
    void resetHistory();

    const ContextHandle context;

    bool reset_history = false;
    uint32_t shown_history_size = 128;

    uint32_t offset;
    PlottingType plotting_type = PlottingType::FLOAT_32;

    bool log_x_axis = false, log_y_axis = false;

    uint32_t skip_interval = 1, skip_counter;

    float max_value = 1.0f, min_value = 0.0f;
    bool auto_find_max = true, auto_find_min = true;
    float test_value;

    std::vector<float> history;
    uint32_t current_history_idx;

    PtrInHandle<const void*> con_src = PtrIn<const void*>::create("src");
};

} // namespace merian_nodes
