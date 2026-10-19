#include "merian-graph/graph/graph.hpp"

#include <algorithm>
#include <unordered_set>

namespace merian {

void Graph::profiler_set_enabled([[maybe_unused]] const bool enabled) {
#ifdef MERIAN_PROFILER_ENABLE
    profiler_enable = enabled;
#endif
}

bool Graph::profiler_get_enabled() const {
    return profiler_enable;
}

ProfilerReport Graph::profiler_get_report(const ProfilerReportKind kind) const {
    switch (kind) {
    case ProfilerReportKind::RUN:
        return profiler_enable ? run_profiler->get_report(Profiler::Period::TOTAL, 0)
                               : ProfilerReport{};
    case ProfilerReportKind::INTERVAL:
        return interval_report;
    case ProfilerReportKind::BUILD:
        return build_report;
    }
    return {};
}

void Graph::profiler_restart_report() {
    run_profiler->clear(Profiler::Period::TOTAL);
}

GraphMemoryReport Graph::memory_get_report() const {
    GraphMemoryReport report;
    report.layer_count = static_cast<uint32_t>(layers.size());
    report.period = period;
    report.aliasing = settings.alias_transient_resources;

    // in run order, then the nodes that do not run
    std::vector<NodeHandle> nodes;
    if (!needs_reconnect) {
        for (const Layer& layer : layers) {
            nodes.insert(nodes.end(), layer.nodes.begin(), layer.nodes.end());
        }
    }
    const std::unordered_set<NodeHandle> running(nodes.begin(), nodes.end());
    for (const auto& [identifier, node] : node_for_identifier) {
        if (!running.contains(node)) {
            nodes.push_back(node);
        }
    }

    for (const NodeHandle& node : nodes) {
        const NodeData& data = node_data.at(node);
        const std::string& identifier = data.identifier;
        report.node_allocations.emplace_back(identifier, data.memory_tracker->get_allocated_size());

        // a pending reconnect may have left the outputs half allocated
        if (needs_reconnect) {
            continue;
        }
        for (const auto& [output, per_output_info] : data.output_connections) {
            // versions written in place share the resources of their root
            if (!per_output_info.resource_data || per_output_info.version_of) {
                continue;
            }
            const std::vector<ResourceData::Copy>& copies = per_output_info.resource_data->copies;
            const OutputConnectorDescriptor& descriptor = data.output_descriptors.at(output);
            for (uint32_t copy_index = 0; copy_index < copies.size(); copy_index++) {
                const TransientMemoryScopeHandle& scope = copies[copy_index].scope;
                const std::string name = graph_internal::copy_name(identifier, descriptor.name,
                                                                   copy_index, copies.size());
                const std::vector<TransientMemoryAllocator::Placement> placements =
                    scope->get_placements();
                for (uint32_t resource_index = 0; resource_index < placements.size();
                     resource_index++) {
                    report.transient.emplace_back(
                        identifier,
                        placements.size() == 1 ? name
                                               : fmt::format("{} [{}]", name, resource_index),
                        descriptor.scratch, descriptor.persistent, placements[resource_index],
                        scope->get_lifetime());
                }
                for (const TransientMemoryAllocator::Fallback& fallback : scope->get_fallbacks()) {
                    report.exclusive.emplace_back(identifier, name, descriptor.scratch, fallback);
                }
            }
        }
    }
    if (!needs_reconnect) {
        for (const TransientMemoryAllocator::Block& block : transient_blocks) {
            report.blocks.emplace_back(block.memory_type_index, block.size, block.requested_size,
                                       block.memory);
        }
        for (const Layer& layer : layers) {
            std::vector<std::string>& labels = report.layers.emplace_back();
            for (const NodeHandle& node : layer.nodes) {
                labels.push_back(node_data.at(node).identifier);
            }
        }
        report.in_use.assign(static_cast<size_t>(period) * layers.size(), 0);
        for (const GraphMemoryReport::Transient& resource : report.transient) {
            for (uint32_t position = 0; position < report.in_use.size(); position++) {
                if (resource.lifetime.contains(position)) {
                    report.in_use[position] += resource.placement.size;
                }
            }
        }
    }

    std::ranges::sort(report.transient, [](const auto& a, const auto& b) {
        return std::tie(a.placement.block_index, a.placement.offset, a.placement.size, a.output) <
               std::tie(b.placement.block_index, b.placement.offset, b.placement.size, b.output);
    });
    std::ranges::sort(report.exclusive, [](const auto& a, const auto& b) {
        return std::tie(b.fallback.size, a.output) < std::tie(a.fallback.size, b.output);
    });

    return report;
}

} // namespace merian
