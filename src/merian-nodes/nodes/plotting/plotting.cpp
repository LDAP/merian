#include "merian-nodes/nodes/plotting/plotting.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "merian-nodes/graph/errors.hpp"

#include <fmt/chrono.h>
#include <iostream>

namespace merian_nodes {

Plotting::Plotting(const ContextHandle context) : Node(), context(context) {

}

Plotting::~Plotting() {}

std::vector<InputConnectorHandle> Plotting::describe_inputs() {
    return {con_src};
}

std::vector<OutputConnectorHandle> Plotting::describe_outputs(const NodeIOLayout& io_layout) {
    return {con_out};
}

Node::NodeStatusFlags Plotting::properties(Properties& config) {
    return {};
}


void Plotting::process([[maybe_unused]] GraphRun& run,
                           const vk::CommandBuffer& cmd,
                           const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    if (io[con_src] != nullptr) {
        io[con_out] = io[con_src];
    }
}

} // namespace merian_nodes
