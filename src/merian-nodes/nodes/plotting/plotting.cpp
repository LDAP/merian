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
    return {};
}

Node::NodeStatusFlags Plotting::properties(Properties& config) {
    if (config.config_bool("Reset history", reset_history, "Resets the history, acts like a button.")) {
        resetHistory();
    }

    if (config.config_uint("History size", shown_history_size, "Size of the shown history.")) {
        if (!log_x_axis) {
            history.resize(shown_history_size * 2);
            current_history_idx = current_history_idx % shown_history_size;
        }
    }

    config.config_uint("Offset", offset, "Offset of the element to plot in byte.");
    //config.config_enum("Element type", plotting_type, Properties::OptionsStyle::DONT_CARE, "Type of the elements to plot.");

    if (config.config_bool("Log(x)", log_x_axis, "Show plot with a logarithmic x-axis.")) {
        resetHistory();
    }
    if (config.config_bool("Log(y)", log_y_axis, "Show plot with a logarithmic y-axis.")) {
        resetHistory();
    }

    config.config_bool("Auto find max", auto_find_max, "Whether or not to find max value automatically.");
    config.config_float("Max Value", max_value, "Max value of the plot", 0.001f);

    if (log_x_axis) {
        config.output_plot_line("", history.data(),
            history.size(), 0, max_value);
    } else {
        config.output_plot_line("", history.data() + current_history_idx + 1,
                (history.size() / 2) - 1, 0, max_value);
    }

    return {};
}

float findMaxInHistory(std::vector<float>& history, size_t size, size_t offset) {
    float max_value = 0;
    for (int i = 0; i < size; i++) {
        if (history[offset + i] > max_value) {
            max_value = history[offset + i];
        }
    }
    return max_value;
}

void Plotting::process([[maybe_unused]] GraphRun& run,
                           const vk::CommandBuffer& cmd,
                           const DescriptorSetHandle& descriptor_set,
                           const NodeIO& io) {
    if (io[con_src] != nullptr) {
        const uint32_t half_size = history.size() / 2;

        if (auto_find_max) {
            max_value = findMaxInHistory(history, half_size, current_history_idx);
        }

        if (!log_x_axis || skip_counter == skip_interval) {
            float value = *static_cast<const float*>(*io[con_src] + offset);
            if (log_y_axis) {
                value = std::log(value);
            }

            if (log_x_axis) {
                // with log x-axis the history grow indefinite
                history.push_back(value);
                skip_interval *= 2;
                skip_counter = 0;
            } else {
                // with linear x-axis, the history wraps around with the configured size
                history[current_history_idx] =
                    history[current_history_idx + half_size] = value;
                current_history_idx = (current_history_idx + 1) % half_size;
            }
        } else {
            skip_counter++;
        }
    }
}

void Plotting::resetHistory() {
    history.clear();
    if (log_x_axis) {
        history.resize(0);
        skip_interval = 1;
        skip_counter = 0;
    } else {
        history.resize(shown_history_size * 2);
        current_history_idx = 0;
    }

    reset_history = false;
}


} // namespace merian_nodes
