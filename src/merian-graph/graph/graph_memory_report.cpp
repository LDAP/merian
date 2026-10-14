#include "merian-graph/graph/graph_memory_report.hpp"
#include "merian/utils/string.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>

namespace merian {

namespace {

std::string_view reason_name(const TransientMemoryAllocator::Fallback::Reason reason) {
    switch (reason) {
    case TransientMemoryAllocator::Fallback::Reason::HOST_ACCESS:
        return "host access";
    case TransientMemoryAllocator::Fallback::Reason::RAW_MEMORY:
        return "raw memory";
    case TransientMemoryAllocator::Fallback::Reason::OWN_MEMORY:
        return "own memory";
    case TransientMemoryAllocator::Fallback::Reason::UNKNOWN_MEMORY_TYPE:
        return "unknown memory type";
    }
    return "";
}

constexpr std::string_view LIFETIME_HEADER = "lifetime";

size_t lifetime_width(const GraphMemoryReport& report) {
    return std::max<size_t>((report.period * (report.layer_count + 1)) - 1, LIFETIME_HEADER.size());
}

// one character per layer, # where in use, | between iterations
std::string lifetime_text(const GraphMemoryReport& report,
                          const TransientMemoryAllocator::Lifetime& lifetime) {
    std::string text;
    for (uint32_t position = 0; position < lifetime.get_positions(); position++) {
        if (position > 0 && position % report.layer_count == 0) {
            text += '|';
        }
        text += lifetime.contains(position) ? '#' : '.';
    }
    return text;
}

std::string placed_line(const GraphMemoryReport& report, const GraphMemoryReport::Placed& placed) {
    return fmt::format("{:<11}{:<11}{:<{}} {}{}\n", format_size(placed.placement.offset),
                       format_size(placed.placement.size), lifetime_text(report, placed.lifetime),
                       lifetime_width(report), placed.output, placed.scratch ? " (scratch)" : "");
}

std::string placed_header(const GraphMemoryReport& report) {
    return fmt::format("{:<11}{:<11}{:<{}} {}\n", "offset", "size", LIFETIME_HEADER,
                       lifetime_width(report), "output");
}

} // namespace

std::string GraphMemoryReport::format_summary() const {
    vk::DeviceSize requested_size = 0;
    vk::DeviceSize block_size = 0;
    for (const Block& block : blocks) {
        requested_size += block.requested_size;
        block_size += block.size;
    }
    vk::DeviceSize unplaced_size = 0;
    for (const Unplaced& entry : unplaced) {
        unplaced_size += entry.fallback.size;
    }
    vk::DeviceSize node_size = 0;
    for (const NodeAllocation& node : node_allocations) {
        node_size += node.size;
    }

    std::string text =
        fmt::format("Outputs: {} placed in {} ({} block(s), aliasing {}, period {})\n",
                    format_size(requested_size), format_size(block_size), blocks.size(),
                    aliasing ? "on" : "off", period);
    if (const auto most_in_use = std::ranges::max_element(in_use); most_in_use != in_use.end()) {
        const auto position = static_cast<uint32_t>(most_in_use - in_use.begin());
        text +=
            fmt::format("  most in use: {} in iteration {}, layer {}\n", format_size(*most_in_use),
                        position / layer_count, position % layer_count);
    }
    text += fmt::format("  unplaced: {}\nNodes: {} (with the unplaced outputs)\n",
                        format_size(unplaced_size), format_size(node_size));
    return text;
}

std::string GraphMemoryReport::format_in_use() const {
    const auto most_in_use = std::ranges::max_element(in_use);
    if (most_in_use == in_use.end()) {
        return {};
    }
    constexpr size_t bar_width = 20;
    std::string text = fmt::format("{:<11}{:<7}{:<11}{:<{}} {}\n", "iteration", "layer", "size", "",
                                   bar_width, "nodes");
    for (uint32_t position = 0; position < in_use.size(); position++) {
        const size_t bar = *most_in_use == 0 ? 0 : in_use[position] * bar_width / *most_in_use;
        const uint32_t layer = position % layer_count;
        text += fmt::format("{:<11}{:<7}{:<11}{:<{}} {}\n", position / layer_count, layer,
                            format_size(in_use[position]), std::string(bar, '#'), bar_width,
                            fmt::join(layers[layer], ", "));
    }
    return text;
}

std::string GraphMemoryReport::format_blocks() const {
    std::string text;
    for (uint32_t block_index = 0; block_index < blocks.size(); block_index++) {
        const Block& block = blocks[block_index];
        text += fmt::format("{}Block {} (memory type {}): {}, {} placed\n",
                            block_index == 0 ? "" : "\n", block_index, block.memory_type_index,
                            format_size(block.size), format_size(block.requested_size));
        text += placed_header(*this);
        for (const Placed& entry : placed) {
            if (entry.placement.block_index == block_index) {
                text += placed_line(*this, entry);
            }
        }
    }
    return text;
}

std::string GraphMemoryReport::format_node(const std::string& node) const {
    std::string text;
    for (const NodeAllocation& allocation : node_allocations) {
        if (allocation.node == node) {
            text += fmt::format("Allocated: {}\n", format_size(allocation.size));
        }
    }

    std::string placed_text;
    for (const Placed& entry : placed) {
        if (entry.node == node) {
            placed_text +=
                fmt::format("{:<7}", entry.placement.block_index) + placed_line(*this, entry);
        }
    }
    if (!placed_text.empty()) {
        text += fmt::format("Outputs: {} placed\n{:<7}{}{}", format_size(placed_size(node)),
                            "block", placed_header(*this), placed_text);
    }

    for (const Unplaced& entry : unplaced) {
        if (entry.node == node) {
            text += fmt::format("Unplaced ({}): {} {}{}\n", reason_name(entry.fallback.reason),
                                format_size(entry.fallback.size), entry.output,
                                entry.scratch ? " (scratch)" : "");
        }
    }
    return text;
}

vk::DeviceSize GraphMemoryReport::placed_size(const std::string& node) const {
    vk::DeviceSize size = 0;
    for (const Placed& entry : placed) {
        if (entry.node == node) {
            size += entry.placement.size;
        }
    }
    return size;
}

std::string format_as(const GraphMemoryReport& report) {
    std::string text = report.format_summary();
    if (const std::string in_use = report.format_in_use(); !in_use.empty()) {
        text += fmt::format("\nIn use\n{}", in_use);
    }
    if (const std::string blocks = report.format_blocks(); !blocks.empty()) {
        text += fmt::format("\n{}", blocks);
    }
    text += "\nNodes";
    for (const GraphMemoryReport::NodeAllocation& node : report.node_allocations) {
        text += fmt::format("\n{}\n{}", node.label, report.format_node(node.node));
    }
    return text;
}

} // namespace merian
