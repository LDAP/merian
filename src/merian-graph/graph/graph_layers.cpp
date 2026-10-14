#include "merian-graph/graph/graph.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace merian {

namespace {
// Predecessors first; misses the nodes on cycles.
std::vector<NodeHandle>
order_nodes(const std::vector<NodeHandle>& nodes,
            const std::unordered_map<NodeHandle, std::vector<NodeHandle>>& predecessors) {
    std::unordered_map<NodeHandle, uint32_t> waiting;
    std::unordered_map<NodeHandle, std::vector<NodeHandle>> successors;
    for (const auto& [node, node_predecessors] : predecessors) {
        waiting[node] += static_cast<uint32_t>(node_predecessors.size());
        for (const NodeHandle& predecessor : node_predecessors) {
            successors[predecessor].push_back(node);
        }
    }
    std::vector<NodeHandle> order;
    std::ranges::copy_if(nodes, std::back_inserter(order),
                         [&](const NodeHandle& node) { return !waiting.contains(node); });
    for (size_t i = 0; i < order.size(); i++) {
        for (const NodeHandle& successor : successors[order[i]]) {
            if (--waiting.at(successor) == 0) {
                order.push_back(successor);
            }
        }
    }
    return order;
}
} // namespace

std::unordered_map<NodeHandle, std::vector<NodeHandle>>
Graph::compute_predecessors(const std::vector<NodeHandle>& nodes) const {
    std::unordered_map<NodeHandle, std::vector<NodeHandle>> predecessors;
    for (const NodeHandle& node : nodes) {
        const NodeData& data = node_data.at(node);
        for (const auto& [input, per_input_info] : data.input_connections) {
            if (per_input_info.node && per_input_info.node != node &&
                data.input_descriptors.at(input).delay == 0) {
                predecessors[node].push_back(per_input_info.node);
            }
        }
        for (const auto& [output, per_output_info] : data.output_connections) {
            for (const auto& [writer, writer_input] : per_output_info.inputs) {
                if (!node_data.at(writer).input_descriptors.at(writer_input).in_place) {
                    continue;
                }
                for (const auto& [reader, reader_input] : per_output_info.inputs) {
                    if (reader != writer &&
                        graph_internal::reads_before_writer(
                            node_data.at(reader).input_descriptors.at(reader_input))) {
                        predecessors[writer].push_back(reader);
                    }
                }
            }
        }
    }
    return predecessors;
}

void Graph::plan_memory() {
    const uint32_t layer_count = static_cast<uint32_t>(layers.size());
    const uint32_t positions = period * layer_count;

    std::vector<MemoryAllocationHandle> reusable;
    for (TransientMemoryAllocator::Block& block : transient_blocks) {
        reusable.push_back(std::move(block.memory));
    }
    transient_blocks.clear();
    const TransientMemoryAllocatorHandle transient_allocator =
        TransientMemoryAllocator::create(resource_allocator->get_memory_allocator());

    // 1. the copies of every output, in declaration order: placements are deterministic
    for (const auto& layer : layers)
        for (const auto& node : layer.nodes) {
            NodeData& data = node_data.at(node);
            for (const OutputConnectorHandle& output : data.output_connectors) {
                NodeData::PerOutputInfo& per_output_info = data.output_connections.at(output);
                const OutputConnectorDescriptor& descriptor = data.output_descriptors.at(output);
                // versions written in place get the resources of their root
                if (descriptor.disabled || per_output_info.version_of) {
                    continue;
                }

                // the output and the versions written in place after it, with all their readers
                std::vector<NodeData::PerOutputInfo*> chain{&per_output_info};
                std::vector<std::tuple<NodeHandle, InputConnectorHandle>> inputs;
                for (size_t i = 0; i < chain.size(); i++) {
                    for (const auto& [reader, input] : chain[i]->inputs) {
                        inputs.emplace_back(reader, input);
                        NodeData& reader_data = node_data.at(reader);
                        // every version of the chain is keyed by the root's connector
                        if (reader_data.input_descriptors.at(input).in_place) {
                            chain.push_back(&reader_data.output_connections.at(output));
                        }
                    }
                }

                // positions from the write to the last read
                uint32_t span = 1;
                uint32_t max_delay = 0;
                ConnectorAccess combined_access = descriptor.access;
                for (const auto& [input_node, input] : inputs) {
                    const NodeData& input_data = node_data.at(input_node);
                    const InputConnectorDescriptor& input_descriptor =
                        input_data.input_descriptors.at(input);
                    span = std::max(span, (input_descriptor.delay * layer_count) +
                                              input_data.level - data.level + 1);
                    max_delay = std::max(max_delay, input_descriptor.delay);
                    combined_access = combined_access | input_descriptor.access;
                }
                const bool always = descriptor.persistent || !alias_transient_resources;
                // without sharing, only the copies the delayed inputs read
                const uint32_t copy_count = descriptor.persistent || !output->supports_delay() ? 1
                                            : always ? max_delay + 1
                                                     : period;

                SPDLOG_DEBUG("creating {} copies of output {} of {}", copy_count, descriptor.name,
                             data.label);

                per_output_info.resource_data = std::make_shared<ResourceData>();
                ResourceData& resource_data = *per_output_info.resource_data;
                for (uint32_t copy_index = 0; copy_index < copy_count; copy_index++) {
                    TransientMemoryAllocator::Lifetime lifetime(positions, always);
                    for (uint32_t iteration = copy_index; !always && iteration < period;
                         iteration += copy_count) {
                        lifetime.add((iteration * layer_count) + data.level, span);
                    }
                    const TransientMemoryScopeHandle scope =
                        transient_allocator->create_scope(lifetime, data.memory_tracker);

                    resource_data.copies.push_back({
                        .resource = output->create_resource({
                            .inputs = inputs,
                            .combined_access = combined_access,
                            .allocator =
                                std::make_shared<ResourceAllocator>(resource_allocator, scope),
                            .persistent = descriptor.persistent,
                            .copy_index = copy_index,
                            .copy_count = copy_count,
                            .debug_name =
                                copy_count == 1
                                    ? fmt::format("{}.{}", data.identifier, descriptor.name)
                                    : fmt::format("{}.{} ({})", data.identifier, descriptor.name,
                                                  copy_index),
                        }),
                        .scope = scope,
                        .read_before_written =
                            copy_count > 1 && copy_index + max_delay >= copy_count,
                    });
                }
                for (NodeData::PerOutputInfo* version : chain) {
                    version->resource_data = per_output_info.resource_data;
                }
            }
        }

    // 2. memory
    transient_allocator->commit(std::move(reusable));
    transient_blocks = transient_allocator->get_blocks();

    // 3. a full barrier and image discards where memory changes hands
    constexpr vk::MemoryBarrier2 full_barrier{
        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eAllCommands,
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
    for (auto& layer : layers)
        for (const auto& node : layer.nodes) {
            for (const auto& [output, per_output_info] : node_data.at(node).output_connections) {
                if (!per_output_info.resource_data || per_output_info.version_of) {
                    continue;
                }
                const std::vector<ResourceData::Copy>& copies =
                    per_output_info.resource_data->copies;
                for (uint32_t copy_index = 0; copy_index < copies.size(); copy_index++) {
                    const ResourceData::Copy& copy = copies[copy_index];
                    if (!copy.aliased()) {
                        continue;
                    }
                    const std::vector<ImageHandle> images = copy.scope->get_aliased_images();
                    for (uint32_t iteration = copy_index; iteration < period;
                         iteration += copies.size()) {
                        Layer::Set& set = layer.sets[iteration];
                        set.barrier.srcStageMask |= full_barrier.srcStageMask;
                        set.barrier.srcAccessMask |= full_barrier.srcAccessMask;
                        set.barrier.dstStageMask |= full_barrier.dstStageMask;
                        set.barrier.dstAccessMask |= full_barrier.dstAccessMask;
                        set.discards.insert(set.discards.end(), images.begin(), images.end());
                    }
                }
            }
        }
}

void Graph::precompute_resources() {
    for (const auto& layer : layers)
        for (const auto& dst_node : layer.nodes) {
            auto& dst_data = node_data.at(dst_node);

            for (uint32_t set_index = 0; set_index < period; set_index++) {
                for (auto& [input, per_input_info] : dst_data.input_connections) {
                    if (!per_input_info.node) {
                        per_input_info.precomputed_resources.emplace_back(nullptr, -1ul);
                        continue;
                    }
                    const NodeData& src_data = node_data.at(per_input_info.node);
                    assert(src_data.errors.empty());
                    assert(src_data.enabled && !src_data.unsupported);
                    const std::vector<ResourceData::Copy>& copies =
                        src_data.output_connections.at(per_input_info.output).resource_data->copies;
                    // the copy written delay iterations ago
                    const uint32_t copy_index =
                        (set_index + period - dst_data.input_descriptors.at(input).delay) %
                        copies.size();
                    per_input_info.precomputed_resources.emplace_back(copies[copy_index].resource,
                                                                      copy_index);
                }
                for (auto& [_, per_output_info] : dst_data.output_connections) {
                    if (!per_output_info.resource_data) {
                        per_output_info.precomputed_resources.emplace_back(nullptr, -1ul);
                        continue;
                    }
                    const std::vector<ResourceData::Copy>& copies =
                        per_output_info.resource_data->copies;
                    const uint32_t copy_index = set_index % copies.size();
                    per_output_info.precomputed_resources.emplace_back(copies[copy_index].resource,
                                                                       copy_index);
                }

                dst_data.resource_maps.emplace_back(this, &dst_data, dst_node, set_index);
            }
        }
}

void Graph::build_layers(const std::vector<NodeHandle>& nodes) {
    const std::unordered_map<NodeHandle, std::vector<NodeHandle>> predecessors =
        compute_predecessors(nodes);
    const std::vector<NodeHandle> order = order_nodes(nodes, predecessors);
    assert(order.size() == nodes.size() && "made sure in validate_in_place_writes");

    uint32_t layer_count = 0;
    for (const NodeHandle& node : order) {
        NodeData& data = node_data.at(node);
        data.level = 0;
        if (const auto it = predecessors.find(node); it != predecessors.end()) {
            for (const NodeHandle& predecessor : it->second) {
                data.level = std::max(data.level, node_data.at(predecessor).level + 1);
            }
        }
        layer_count = std::max(layer_count, data.level + 1);
    }

    layers.assign(layer_count, {});
    for (const NodeHandle& node : nodes) {
        layers[node_data.at(node).level].nodes.push_back(node);
    }

    for (auto& layer : layers) {
        std::sort(layer.nodes.begin(), layer.nodes.end(),
                  [&](const NodeHandle& a, const NodeHandle& b) {
                      const NodeData& data_a = node_data.at(a);
                      const NodeData& data_b = node_data.at(b);
                      return std::tie(data_a.linearization_order, data_a.identifier) <
                             std::tie(data_b.linearization_order, data_b.identifier);
                  });
    }

    struct Masks {
        vk::PipelineStageFlags2 stages;
        vk::AccessFlags2 access;
        vk::AccessFlags2 write_access;
    };
    std::vector<Masks> masks(layer_count);
    Masks all{};
    for (uint32_t l = 0; l < layer_count; l++) {
        const auto add = [&](const ConnectorAccess& access) {
            masks[l].stages |= access.stages;
            masks[l].access |= access.access;
            if (access.is_write()) {
                masks[l].write_access |= access.access;
            }
        };
        for (const NodeHandle& node : layers[l].nodes) {
            const NodeData& data = node_data.at(node);
            for (const auto& [input, descriptor] : data.input_descriptors) {
                add(descriptor.access);
            }
            for (const auto& [output, descriptor] : data.output_descriptors) {
                add(descriptor.access);
            }
        }
        all.stages |= masks[l].stages;
        all.access |= masks[l].access;
        all.write_access |= masks[l].write_access;
    }

    period = 1;
    for (const NodeHandle& node : nodes) {
        for (const auto& [input, per_input_info] : node_data.at(node).input_connections) {
            if (per_input_info.node) {
                period = lcm(period, node_data.at(node).input_descriptors.at(input).delay + 1);
            }
        }
    }

    if (layer_count > 0) {
        layers[0].sets.assign(period, {.barrier = vk::MemoryBarrier2{all.stages, all.write_access,
                                                                     all.stages, all.access},
                                       .discards = {}});
    }
    Masks prefix{};
    for (uint32_t l = 1; l < layer_count; l++) {
        prefix.stages |= masks[l - 1].stages;
        prefix.write_access |= masks[l - 1].write_access;
        layers[l].sets.assign(period,
                              {.barrier = vk::MemoryBarrier2{prefix.stages, prefix.write_access,
                                                             masks[l].stages, masks[l].access},
                               .discards = {}});
    }
}

} // namespace merian
