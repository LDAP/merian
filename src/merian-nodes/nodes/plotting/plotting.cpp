#include "merian-nodes/nodes/plotting/plotting.hpp"

#include "merian/vk/pipeline/pipeline_compute.hpp"
#include "merian/vk/pipeline/pipeline_layout_builder.hpp"
#include "merian/vk/pipeline/specialization_info_builder.hpp"

#include "merian-nodes/graph/errors.hpp"

#include <fmt/chrono.h>
#include <iostream>

namespace merian_nodes {

Plotting::Plotting(const ContextHandle context) : Node(), context(context) {
    history = std::vector<float>(shown_history_size * 2);
}

Plotting::~Plotting() {}

std::vector<InputConnectorHandle> Plotting::describe_inputs() {
    return {con_src};
}

std::vector<OutputConnectorHandle> Plotting::describe_outputs(const NodeIOLayout& io_layout) {
    return {con_out};
}

Node::NodeStatusFlags Plotting::properties(Properties& config) {
    config.config_uint("Index to plot", plotting_idx, "Which index of the buffer should be plottet.");

    if (config.config_uint("History size", shown_history_size, "Size of the shown history.")) {
        history.resize(shown_history_size * 2);
        current_history_idx = current_history_idx % shown_history_size;
    }

    config.config_float("Test Value", test_value, "");

    config.config_float("Max Value", max_value, "Max value of the plot");
    config.output_plot_line("", history.data() + current_history_idx + 1,
        (history.size() / 2) - 1, 0, max_value);
    return {};
}


void Plotting::process([[maybe_unused]] GraphRun& run,
                           const vk::CommandBuffer& cmd,
                           const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    if (io[con_src] != nullptr) {
        io[con_out] = io[con_src];

        const uint32_t half_size = history.size() / 2;
        history[current_history_idx] =
            history[current_history_idx + half_size] = (*io[con_src])[plotting_idx].x; // TODO how to decide the component
        current_history_idx = (current_history_idx + 1) % half_size;
    }



    SPDLOG_INFO("{}", plotting_idx);
}

} // namespace merian_nodes
