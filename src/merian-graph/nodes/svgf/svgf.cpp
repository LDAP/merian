#include "merian-graph/nodes/svgf/svgf.hpp"

namespace merian {

Subgraph SVGF::describe_subgraph() {
    Subgraph subgraph;
    subgraph.add_node("variance estimate", variance_estimate);
    subgraph.add_connection(".src", "variance estimate.src");
    subgraph.add_connection(".history", "variance estimate.history");
    subgraph.add_connection(".gbuffer", "variance estimate.gbuffer");

    std::string previous = "variance estimate";
    for (uint32_t level = 0; level < filters.size(); level++) {
        const std::string name = fmt::format("filter {}", level);
        subgraph.add_node(name, filters[level]);
        subgraph.add_connection(previous + ".out", name + ".src");
        subgraph.add_connection(previous + ".gbuf", name + ".gbuf_src");
        subgraph.add_connection(".gbuffer", name + ".gbuffer");
        previous = name;
    }

    subgraph.add_node("taa", taa);
    subgraph.add_connection(previous + ".out", "taa.src");
    subgraph.add_connection(".gbuffer", "taa.gbuffer");
    subgraph.add_connection(".mv", "taa.mv");
    subgraph.add_connection(".prev_out", "taa.prev_out");
    subgraph.add_connection("taa.out", ".out");
    // the source lacks the albedo the TAA re-modulates
    if (!taa->modulates_albedo()) {
        subgraph.add_pass_through(".src", ".out");
    }
    return subgraph;
}

SVGF::NodeStatusFlags SVGF::properties(Properties& config) {
    NodeStatusFlags flags{};

    if (config.config_int("iterations", filter_levels, "0 disables the filter (TAA only)", 0, 10)) {
        set_filter_levels(filter_levels);
        flags |= NEEDS_RECONNECT;
    }
    if (config.st_begin_child("filter", "Filter", Properties::ChildFlagBits::DEFAULT_OPEN)) {
        if (filter_parameters.properties(config)) {
            for (const std::shared_ptr<SVGFFilter>& filter : filters) {
                filter->set_parameters(filter_parameters);
            }
        }
        if (options.properties(config)) {
            variance_estimate->set_options(options);
            for (const std::shared_ptr<SVGFFilter>& filter : filters) {
                filter->set_options(options);
            }
            flags |= NEEDS_RECONNECT;
        }
        if (!filters.empty()) {
            config.output_text("local size: {}", filters.front()->get_local_size());
        }
        config.st_end_child();
    }
    if (config.config_enum("overwrite format", overwrite_format, Properties::OptionsStyle::COMBO,
                           "Undefined keeps the format of the input.")) {
        variance_estimate->set_format(overwrite_format);
        flags |= NEEDS_RECONNECT;
    }

    if (config.st_begin_child("variance estimate", "Variance Estimate")) {
        flags |= variance_estimate->properties(config);
        config.st_end_child();
    }
    if (config.st_begin_child("taa", "TAA")) {
        flags |= taa->properties(config);
        config.st_end_child();
    }
    return flags;
}

void SVGF::set_filter_levels(const uint32_t level_count) {
    filters.resize(level_count);
    for (uint32_t level = 0; level < level_count; level++) {
        if (!filters[level]) {
            filters[level] = std::make_shared<SVGFFilter>();
            filters[level]->set_options(options);
            filters[level]->set_parameters(filter_parameters);
        }
        filters[level]->set_level(level, level_count);
    }
    variance_estimate->set_filter_levels(level_count);
}

} // namespace merian
