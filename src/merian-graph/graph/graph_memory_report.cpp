#include "merian-graph/graph/graph_memory_report.hpp"
#include "merian/utils/string.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>

namespace merian {

namespace {

constexpr std::string_view FULL = "█";
constexpr std::string_view FREE = "·";
constexpr std::string_view PEAK = "▶";
constexpr std::array<std::string_view, 8> LEFT_EIGHTHS = {"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"};

constexpr size_t SIZE_COLUMN_WIDTH = 9;
constexpr size_t LAYER_WIDTH = 3;
constexpr uint32_t GAUGE_WIDTH = 32;
constexpr uint32_t WHERE_WIDTH = 20;
constexpr uint32_t MAP_WIDTH = 32;

std::string_view reason_name(const TransientMemoryAllocator::Fallback::Reason reason) {
    switch (reason) {
    case TransientMemoryAllocator::Fallback::Reason::HOST_ACCESS:
        return "host access";
    case TransientMemoryAllocator::Fallback::Reason::RAW_MEMORY:
        return "raw memory";
    case TransientMemoryAllocator::Fallback::Reason::OWN_MEMORY:
        return "sparse, disjoint or extended";
    case TransientMemoryAllocator::Fallback::Reason::UNKNOWN_MEMORY_TYPE:
        return "unknown memory type";
    }
    return "";
}

size_t text_width(const std::string_view text) {
    return std::ranges::count_if(
        text, [](const char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; });
}

std::string align_left(std::string text, const size_t width) {
    const size_t current = text_width(text);
    if (current < width) {
        text.append(width - current, ' ');
    }
    return text;
}

std::string align_right(const std::string_view text, const size_t width) {
    const size_t current = text_width(text);
    return current < width ? std::string(width - current, ' ').append(text) : std::string(text);
}

std::string repeat(const std::string_view text, const size_t count) {
    std::string repeated;
    repeated.reserve(text.size() * count);
    for (size_t i = 0; i < count; i++) {
        repeated += text;
    }
    return repeated;
}

std::string size_cell(const vk::DeviceSize size, const size_t width = SIZE_COLUMN_WIDTH) {
    return size == 0 ? align_right(FREE, width) : align_right(format_size(size), width);
}

std::string axis(const std::string_view begin, const std::string_view end, const size_t width) {
    return align_left(std::string(begin), width - std::min(width, text_width(end))).append(end);
}

std::string axis(const vk::DeviceSize size, const size_t width) {
    return axis("0 B", format_size(size), width);
}

std::string titled_rule(const std::string_view title, const size_t width) {
    const std::string titled = fmt::format("── {} ", title);
    return titled + repeat("─", width - std::min(width, text_width(titled)));
}

std::string gauge(const double fraction, const uint32_t width) {
    uint32_t eighths =
        static_cast<uint32_t>(std::lround(std::clamp(fraction, 0.0, 1.0) * width * 8));
    if (fraction > 0) {
        eighths = std::max(eighths, 1u);
    }
    std::string text;
    for (uint32_t cell = 0; cell < width; cell++) {
        const uint32_t filled = std::min(eighths - std::min(eighths, cell * 8), 8u);
        text += filled == 8 ? FULL : filled == 0 ? FREE : LEFT_EIGHTHS[filled];
    }
    return text;
}

struct BarPart {
    vk::DeviceSize size;
    std::string_view pattern;
};

std::string stacked_bar(const std::span<const BarPart> parts, const uint32_t width) {
    vk::DeviceSize total = 0;
    size_t largest = 0;
    for (size_t i = 0; i < parts.size(); i++) {
        total += parts[i].size;
        if (parts[i].size > parts[largest].size) {
            largest = i;
        }
    }
    if (total == 0) {
        return repeat(FREE, width);
    }
    std::vector<int64_t> cells;
    int64_t used = 0;
    for (const BarPart& part : parts) {
        const double share = static_cast<double>(part.size) * width / static_cast<double>(total);
        cells.push_back(part.size == 0 ? 0 : std::max<int64_t>(1, std::llround(share)));
        used += cells.back();
    }
    cells[largest] += static_cast<int64_t>(width) - used;
    std::string text;
    for (size_t i = 0; i < parts.size(); i++) {
        text += repeat(parts[i].pattern, cells[i]);
    }
    return text;
}

std::string range_bar(const vk::DeviceSize begin,
                      const vk::DeviceSize end,
                      const vk::DeviceSize size,
                      const uint32_t width) {
    const double scale =
        static_cast<double>(width) / static_cast<double>(std::max<vk::DeviceSize>(size, 1));
    const double range_begin = static_cast<double>(begin) * scale;
    const double range_end = static_cast<double>(end) * scale;
    std::string text;
    for (uint32_t cell = 0; cell < width; cell++) {
        const double covered = std::min(range_end, cell + 1.0) - std::max(range_begin, cell + 0.0);
        const bool from_left = range_begin <= cell + 0.125;
        const bool to_right = range_end >= cell + 0.875;
        if (covered <= 0) {
            text += " ";
        } else if (from_left && to_right) {
            text += FULL;
        } else if (from_left) {
            text += LEFT_EIGHTHS[std::clamp<long>(std::lround(covered * 8), 1, 7)];
        } else if (to_right) {
            text += covered >= 0.375 ? "▐" : "▕";
        } else {
            text += LEFT_EIGHTHS[1];
        }
    }
    return text;
}

std::string lifetime_bar(const uint32_t layer_count,
                         const TransientMemoryAllocator::Lifetime& lifetime) {
    std::string text;
    for (uint32_t position = 0; position < lifetime.get_positions(); position++) {
        if (position > 0 && position % layer_count == 0) {
            text += "│";
        }
        text += lifetime.contains(position) ? FULL : " ";
    }
    return text;
}

std::string lifetime_header(const uint32_t layer_count, const uint32_t period) {
    std::string text;
    for (uint32_t position = 0; position < layer_count * period; position++) {
        if (position > 0 && position % layer_count == 0) {
            text += " ";
        }
        text += static_cast<char>('0' + (position % layer_count % 10));
    }
    return text;
}

std::string layer_nodes(const GraphMemoryReport& report, const uint32_t position) {
    return fmt::format("{}", fmt::join(report.layers[position % report.layer_count], ", "));
}

std::string_view connector_name(const std::string& node, const std::string& output) {
    if (output.size() > node.size() && output.starts_with(node) && output[node.size()] == '.') {
        return std::string_view(output).substr(node.size() + 1);
    }
    return output;
}

std::string_view kind(const bool scratch, const bool persistent) {
    if (scratch) {
        return persistent ? "  persistent scratch" : "  scratch";
    }
    return persistent ? "  persistent" : "";
}

std::unordered_map<std::string_view, std::string> node_prefixes(const GraphMemoryReport& report) {
    std::unordered_set<std::string_view> nodes;
    for (const GraphMemoryReport::Transient& entry : report.transient) {
        nodes.insert(entry.node);
    }
    std::unordered_map<std::string_view, std::string> prefixes;
    for (const std::string_view node : nodes) {
        size_t length = 1;
        while (length < node.size() && std::ranges::any_of(
                                           nodes,
                                           [&](const std::string_view other) {
                                               return other != node &&
                                                      other.starts_with(node.substr(0, length));
                                           })) {
            length++;
        }
        std::string prefix(node.substr(0, length));
        prefix[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(prefix[0])));
        prefixes.emplace(node, std::move(prefix));
    }
    return prefixes;
}

} // namespace

vk::DeviceSize GraphMemoryReport::total_size() const {
    vk::DeviceSize size = 0;
    for (const Block& block : blocks) {
        size += block.size;
    }
    for (const NodeAllocation& node : node_allocations) {
        size += node.size;
    }
    return size;
}

std::string GraphMemoryReport::summary_text() const {
    vk::DeviceSize block_size = 0;
    vk::DeviceSize output_size = 0;
    for (const Block& block : blocks) {
        block_size += block.size;
        output_size += block.requested_size;
    }
    vk::DeviceSize exclusive_size = 0;
    for (const Exclusive& entry : exclusive) {
        exclusive_size += entry.fallback.size;
    }
    vk::DeviceSize node_size = 0;
    for (const NodeAllocation& node : node_allocations) {
        node_size += node.size;
    }
    node_size -= std::min(node_size, exclusive_size);

    constexpr size_t label_width = 20;
    const auto line = [&](const std::string_view label, const vk::DeviceSize size,
                          const std::string_view description) {
        return fmt::format("{}{}  {}\n", align_left(std::string(label), label_width),
                           size_cell(size), description);
    };

    const std::array<BarPart, 3> parts{{
        {node_size, "█"},
        {block_size, "▓"},
        {exclusive_size, "░"},
    }};
    std::string text =
        line("Total", node_size + block_size + exclusive_size, stacked_bar(parts, GAUGE_WIDTH));
    text += line("  nodes", node_size, "█ allocated by the nodes themselves");
    text += line("  output blocks", block_size,
                 aliasing
                     ? fmt::format("▓ shared by {} of outputs", format_size(output_size))
                     : fmt::format("▓ hold {} of outputs, aliasing off", format_size(output_size)));
    text +=
        line("  exclusive outputs", exclusive_size, "░ one allocation each, outside the blocks");
    if (const auto peak = std::ranges::max_element(in_use); peak != in_use.end()) {
        const auto position = static_cast<uint32_t>(peak - in_use.begin());
        const std::string when = period == 1
                                     ? fmt::format("layer {}", position % layer_count)
                                     : fmt::format("iteration {}, layer {}", position / layer_count,
                                                   position % layer_count);
        text += line("Peak in use", *peak,
                     fmt::format("of the blocks, at {} ({})", when, layer_nodes(*this, position)));
    }
    return text;
}

std::string GraphMemoryReport::nodes_text() const {
    enum Column : uint8_t { OWN, TRANSIENT, PERSISTENT, EXCLUSIVE, TOTAL, COLUMN_COUNT };
    constexpr std::array<std::string_view, COLUMN_COUNT> column_names = {
        "own", "transient", "persistent", "exclusive", "total"};
    constexpr size_t column_width = 11;
    struct Row {
        std::string_view node;
        std::array<vk::DeviceSize, COLUMN_COUNT> sizes{};
    };

    std::vector<Row> rows;
    std::unordered_map<std::string_view, size_t> row_for_node;
    for (const NodeAllocation& allocation : node_allocations) {
        row_for_node.try_emplace(allocation.node, rows.size());
        rows.push_back({allocation.node, {allocation.size}});
    }
    for (const Transient& entry : transient) {
        rows[row_for_node.at(entry.node)].sizes[entry.persistent ? PERSISTENT : TRANSIENT] +=
            entry.placement.size;
    }
    for (const Exclusive& entry : exclusive) {
        rows[row_for_node.at(entry.node)].sizes[EXCLUSIVE] += entry.fallback.size;
    }

    Row sum{"sum"};
    for (Row& row : rows) {
        row.sizes[OWN] -= std::min(row.sizes[OWN], row.sizes[EXCLUSIVE]);
        row.sizes[TOTAL] =
            row.sizes[OWN] + row.sizes[TRANSIENT] + row.sizes[PERSISTENT] + row.sizes[EXCLUSIVE];
        for (uint32_t column = 0; column < COLUMN_COUNT; column++) {
            sum.sizes[column] += row.sizes[column];
        }
    }
    std::ranges::stable_sort(rows, std::greater{}, [](const Row& row) { return row.sizes[TOTAL]; });

    size_t node_width = std::string_view("node").size();
    for (const Row& row : rows) {
        node_width = std::max(node_width, text_width(row.node));
    }
    const auto format_row = [&](const Row& row) {
        std::string line = align_left(std::string(row.node), node_width);
        for (const vk::DeviceSize size : row.sizes) {
            line += size_cell(size, column_width);
        }
        return line + "\n";
    };

    std::string text = align_left("node", node_width);
    for (const std::string_view name : column_names) {
        text += align_right(name, column_width);
    }
    const std::string rule = repeat("─", node_width + (COLUMN_COUNT * column_width)) + "\n";
    text += "\n" + rule;
    for (const Row& row : rows) {
        text += format_row(row);
    }
    text += rule + format_row(sum);
    text += "\nown         allocated by the node itself\n"
            "transient   its outputs in the blocks, may share memory\n"
            "persistent  its outputs in the blocks, kept the whole run\n"
            "exclusive   its outputs with one allocation each, outside the blocks\n";
    return text;
}

std::string GraphMemoryReport::layers_text() const {
    const auto peak = std::ranges::max_element(in_use);
    if (peak == in_use.end()) {
        return {};
    }
    const auto peak_position = static_cast<uint32_t>(peak - in_use.begin());
    vk::DeviceSize block_size = 0;
    for (const Block& block : blocks) {
        block_size += block.size;
    }

    constexpr size_t label_width = LAYER_WIDTH + 2;
    std::string text =
        fmt::format("{}  {}  {}  nodes\n", align_left("layer", label_width),
                    align_right("in use", SIZE_COLUMN_WIDTH),
                    align_left(fmt::format("of {}", format_size(block_size)), GAUGE_WIDTH));
    for (uint32_t position = 0; position < in_use.size(); position++) {
        const uint32_t layer = position % layer_count;
        if (period > 1 && layer == 0) {
            text += titled_rule(fmt::format("iteration {}", position / layer_count),
                                label_width + SIZE_COLUMN_WIDTH + GAUGE_WIDTH + 4) +
                    "\n";
        }
        const double fraction =
            block_size == 0 ? 0 : static_cast<double>(in_use[position]) / block_size;
        text += fmt::format("{} {:>{}}  {}  {}  {}\n", position == peak_position ? PEAK : " ",
                            layer, LAYER_WIDTH, size_cell(in_use[position]),
                            gauge(fraction, GAUGE_WIDTH), layer_nodes(*this, position));
    }
    return text;
}

std::string GraphMemoryReport::outputs_text() const {
    if (transient.empty() && exclusive.empty()) {
        return {};
    }
    const bool several_blocks = blocks.size() > 1;
    size_t name_width = std::string_view("output").size();
    for (const NodeAllocation& node : node_allocations) {
        name_width = std::max(name_width, text_width(node.node));
    }
    for (const Transient& entry : transient) {
        name_width = std::max(name_width, 2 + text_width(connector_name(entry.node, entry.output)));
    }
    for (const Exclusive& entry : exclusive) {
        name_width = std::max(name_width, 2 + text_width(connector_name(entry.node, entry.output)));
    }
    const auto row = [&](const std::string_view name, const vk::DeviceSize size) {
        return fmt::format("  {} {}  ", align_left(std::string(name), name_width - 2),
                           size_cell(size));
    };

    std::string text = fmt::format("{} {}  ", align_left("output", name_width),
                                   align_right("size", SIZE_COLUMN_WIDTH));
    if (!blocks.empty()) {
        text += several_blocks ? "block  " + axis("start", "end", WHERE_WIDTH)
                               : " " + axis(blocks.front().size, WHERE_WIDTH);
        text += " " + lifetime_header(layer_count, period);
    }
    text += "\n";

    for (const NodeAllocation& node : node_allocations) {
        std::vector<const Transient*> node_transient;
        for (const Transient& entry : transient) {
            if (entry.node == node.node) {
                node_transient.push_back(&entry);
            }
        }
        std::vector<const Exclusive*> node_exclusive;
        for (const Exclusive& entry : exclusive) {
            if (entry.node == node.node) {
                node_exclusive.push_back(&entry);
            }
        }
        if (node_transient.empty() && node_exclusive.empty()) {
            continue;
        }
        std::ranges::sort(node_transient, [](const Transient* a, const Transient* b) {
            return std::tie(a->placement.block_index, a->output) <
                   std::tie(b->placement.block_index, b->output);
        });

        text += node.node + "\n";
        for (const Transient* entry : node_transient) {
            const TransientMemoryAllocator::Placement& placement = entry->placement;
            text += row(connector_name(entry->node, entry->output), placement.size);
            if (several_blocks) {
                text += fmt::format("{:>5}  ", placement.block_index);
            }
            text += fmt::format("│{}│{}│{}\n",
                                range_bar(placement.offset, placement.offset + placement.size,
                                          blocks[placement.block_index].size, WHERE_WIDTH),
                                lifetime_bar(layer_count, entry->lifetime),
                                kind(entry->scratch, entry->persistent));
        }
        for (const Exclusive* entry : node_exclusive) {
            text += row(connector_name(entry->node, entry->output), entry->fallback.size);
            text += fmt::format("exclusive · {}{}\n", reason_name(entry->fallback.reason),
                                kind(entry->scratch, false));
        }
    }
    return text;
}

std::string GraphMemoryReport::blocks_text() const {
    if (layer_count == 0) {
        return {};
    }
    const std::string indent(LAYER_WIDTH + 3, ' ');
    const std::unordered_map<std::string_view, std::string> prefixes = node_prefixes(*this);

    std::string text;
    for (uint32_t block_index = 0; block_index < blocks.size(); block_index++) {
        const Block& block = blocks[block_index];
        std::vector<const Transient*> entries;
        for (const Transient& entry : transient) {
            if (entry.placement.block_index == block_index) {
                entries.push_back(&entry);
            }
        }
        const auto cell_begin = [&](const uint32_t cell) { return block.size * cell / MAP_WIDTH; };
        const auto cell_of = [&](const vk::DeviceSize offset) {
            return static_cast<uint32_t>(std::min<vk::DeviceSize>(
                offset * MAP_WIDTH / std::max<vk::DeviceSize>(block.size, 1), MAP_WIDTH - 1));
        };

        text += fmt::format("{}Block {} · {} · memory type {} · {} per column\n",
                            block_index == 0 ? "" : "\n", block_index, format_size(block.size),
                            block.memory_type_index, format_size(cell_begin(1)));
        text += fmt::format("{}{}  {}  nodes\n", align_left("layer", indent.size() + 1),
                            axis(block.size, MAP_WIDTH), align_right("in use", SIZE_COLUMN_WIDTH));

        // 1. rows, a cell shows the output covering most of it
        std::vector<std::string> rows;
        std::vector<vk::DeviceSize> row_in_use;
        for (uint32_t position = 0; position < in_use.size(); position++) {
            std::vector<std::optional<uint32_t>> covering(MAP_WIDTH);
            std::vector<vk::DeviceSize> most_covered(MAP_WIDTH, 0);
            vk::DeviceSize alive_size = 0;
            for (uint32_t index = 0; index < entries.size(); index++) {
                const Transient& entry = *entries[index];
                if (!entry.lifetime.contains(position)) {
                    continue;
                }
                alive_size += entry.placement.size;
                const vk::DeviceSize end = entry.placement.offset + entry.placement.size;
                for (uint32_t cell = cell_of(entry.placement.offset);
                     cell < MAP_WIDTH && cell_begin(cell) < end; cell++) {
                    const vk::DeviceSize covered =
                        std::min(end, cell_begin(cell + 1)) -
                        std::max(entry.placement.offset, cell_begin(cell));
                    if (covered > most_covered[cell]) {
                        most_covered[cell] = covered;
                        covering[cell] = index;
                    }
                }
            }
            std::string cells;
            for (uint32_t cell = 0; cell < MAP_WIDTH;) {
                uint32_t run_end = cell + 1;
                while (run_end < MAP_WIDTH && covering[run_end] == covering[cell]) {
                    run_end++;
                }
                const uint32_t length = run_end - cell;
                if (covering[cell]) {
                    const std::string& prefix = prefixes.at(entries[*covering[cell]]->node);
                    const size_t shown = std::min<size_t>(length, prefix.size());
                    cells += prefix.substr(0, shown) + repeat("─", length - shown);
                } else {
                    cells += std::string(length, ' ');
                }
                cell = run_end;
            }
            rows.push_back(std::move(cells));
            row_in_use.push_back(alive_size);
        }
        const auto peak =
            static_cast<uint32_t>(std::ranges::max_element(row_in_use) - row_in_use.begin());
        for (uint32_t position = 0; position < rows.size(); position++) {
            const uint32_t layer = position % layer_count;
            if (layer == 0) {
                const std::string top =
                    period == 1 ? repeat("─", MAP_WIDTH)
                                : titled_rule(fmt::format("iteration {}", position / layer_count),
                                              MAP_WIDTH);
                text += fmt::format("{}{}{}{}\n", indent, position == 0 ? "┌" : "├", top,
                                    position == 0 ? "┐" : "┤");
            }
            text += fmt::format("{} {:>{}} │{}│ {}  {}\n", position == peak ? PEAK : " ", layer,
                                LAYER_WIDTH, rows[position], size_cell(row_in_use[position]),
                                layer_nodes(*this, position));
        }

        // 2. labels below the cells the outputs start in
        std::vector<std::vector<uint32_t>> entries_at_cell(MAP_WIDTH);
        for (uint32_t index = 0; index < entries.size(); index++) {
            entries_at_cell[cell_of(entries[index]->placement.offset)].push_back(index);
        }
        std::string bottom;
        for (const std::vector<uint32_t>& at_cell : entries_at_cell) {
            bottom += at_cell.empty() ? "─" : "┬";
        }
        text += fmt::format("{}└{}┘\n", indent, bottom);
        for (uint32_t labeled = MAP_WIDTH; labeled-- > 0;) {
            const std::vector<uint32_t>& at_cell = entries_at_cell[labeled];
            for (uint32_t i = 0; i < at_cell.size(); i++) {
                const Transient& entry = *entries[at_cell[i]];
                std::string line = indent + " ";
                for (uint32_t cell = 0; cell < labeled; cell++) {
                    line += entries_at_cell[cell].empty() ? " " : "│";
                }
                text += fmt::format("{}{}─▶ {}  {}\n", line, i + 1 < at_cell.size() ? "├" : "└",
                                    entry.output, format_size(entry.placement.size));
            }
        }
    }
    return text;
}

std::string GraphMemoryReport::to_text(const View view) const {
    switch (view) {
    case View::SUMMARY:
        return summary_text();
    case View::NODES:
        return nodes_text();
    case View::LAYERS:
        return layers_text();
    case View::OUTPUTS:
        return outputs_text();
    case View::BLOCKS:
        return blocks_text();
    }
    return {};
}

std::string GraphMemoryReport::to_text() const {
    std::string text = summary_text();
    const auto section = [&](const std::string_view title, const std::string& body) {
        if (!body.empty()) {
            text += fmt::format("\n{}\n{}", title, body);
        }
    };
    section("Nodes", nodes_text());
    section("Layers", layers_text());
    section("Outputs", outputs_text());
    section("Blocks", blocks_text());
    return text;
}

nlohmann::json GraphMemoryReport::to_json() const {
    const auto positions = [](const TransientMemoryAllocator::Lifetime& lifetime) {
        nlohmann::json in_use = nlohmann::json::array();
        for (uint32_t position = 0; position < lifetime.get_positions(); position++) {
            if (lifetime.contains(position)) {
                in_use.push_back(position);
            }
        }
        return in_use;
    };

    nlohmann::json json = {
        {"total_size", total_size()}, {"layer_count", layer_count}, {"period", period},
        {"aliasing", aliasing},       {"layers", layers},           {"in_use", in_use},
    };
    nlohmann::json& json_blocks = json["blocks"] = nlohmann::json::array();
    for (const Block& block : blocks) {
        json_blocks.push_back({{"memory_type_index", block.memory_type_index},
                               {"size", block.size},
                               {"requested_size", block.requested_size}});
    }
    nlohmann::json& json_transient = json["transient"] = nlohmann::json::array();
    for (const Transient& entry : transient) {
        json_transient.push_back({{"node", entry.node},
                                  {"output", entry.output},
                                  {"scratch", entry.scratch},
                                  {"persistent", entry.persistent},
                                  {"block", entry.placement.block_index},
                                  {"offset", entry.placement.offset},
                                  {"size", entry.placement.size},
                                  {"aliased", entry.placement.aliased},
                                  {"in_use", positions(entry.lifetime)}});
    }
    nlohmann::json& json_exclusive = json["exclusive"] = nlohmann::json::array();
    for (const Exclusive& entry : exclusive) {
        json_exclusive.push_back({{"node", entry.node},
                                  {"output", entry.output},
                                  {"scratch", entry.scratch},
                                  {"size", entry.fallback.size},
                                  {"reason", reason_name(entry.fallback.reason)}});
    }
    nlohmann::json& json_nodes = json["node_allocations"] = nlohmann::json::array();
    for (const NodeAllocation& allocation : node_allocations) {
        json_nodes.push_back({{"node", allocation.node}, {"size", allocation.size}});
    }
    return json;
}

} // namespace merian
