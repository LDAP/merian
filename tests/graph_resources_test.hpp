#pragma once

#include "graph_pattern_model.hpp"
#include "graph_test.hpp"

class GraphResourcesTest : public GraphTest {
  protected:
    uint32_t aliased_checks = 0;
    uint32_t passed_on_records = 0;
    uint32_t in_place_records = 0;
    uint32_t in_place_errors = 0;

    static void SetUpTestSuite() {
        NodeRegistry& registry = NodeRegistry::get_instance();
        registry.register_node_type<PatternNode>("Pattern", "test-graph-resources");
        registry.register_node_type<IntSource>("Int Source", "test-graph-resources");
        registry.register_node_type<CanvasSource>("Canvas Source", "test-graph-resources");
        registry.register_node_type<FixedIntSource>("Fixed Int Source", "test-graph-resources");
        registry.register_node_type<IntAdd>("Int Add", "test-graph-resources");
        registry.register_node_type<IntReader>("Int Reader", "test-graph-resources");
        registry.register_node_type<PatternGroup>("Pattern Group", "test-graph-resources");
        registry.register_node_type<UnsupportedPatternGroup>("Unsupported Pattern Group",
                                                             "test-graph-resources");
        registry.register_node_type<SwapGroup>("Swap Group", "test-graph-resources");
        registry.register_node_type<ScriptedGroup>("Scripted Group", "test-graph-resources");
        registry.register_node_type<ThrowingNode>("Throwing", "test-graph-resources");
        registry.register_node_type<MoveGroup>("Move Group", "test-graph-resources");
        create_context("test-graph-resources");
    }

    static void set_graph_properties(const GraphHandle& graph,
                                     const bool alias,
                                     const uint32_t iterations_in_flight) {
        JSONLoadProperties props(json{{"graph_properties",
                                       {{"alias transient resources", alias},
                                        {"iterations in flight", iterations_in_flight}}}});
        graph->properties(props);
    }

    static GraphHandle load(const GraphSpec& spec) {
        const GraphHandle graph = create_graph(describe(spec, true, 2));
        return graph;
    }

    static void forget_runs(const GraphHandle& graph) {
        for (const std::string& identifier : graph->get_identifiers()) {
            if (const auto node =
                    graph->find_node_for_identifier_and_type<PatternNode>(identifier)) {
                node->records.clear();
                node->process_count = 0;
            }
        }
    }

    void run_and_check(const GraphHandle& graph,
                       const GraphSpec& spec,
                       const uint32_t iterations,
                       const std::string& phase) {
        for (uint32_t i = 0; i < iterations; i++) {
            graph->run();
        }
        graph->wait();
        EXPECT_EQ(stored_connections(graph), stored_connections(describe(spec, true, 1))) << phase;
        check_reads(graph, spec, iterations, phase);
    }

    void run_and_check(const GraphHandle& graph,
                       const GroupedSpec& grouped,
                       const uint32_t iterations,
                       const std::string& phase) {
        for (uint32_t i = 0; i < iterations; i++) {
            graph->run();
        }
        graph->wait();
        const json stored = graph->to_description().to_json();
        const json expected = describe(grouped, true, 1);
        EXPECT_EQ(stored_connections(stored), stored_connections(expected)) << phase;
        EXPECT_EQ(stored.at("nodes").size(), expected.at("nodes").size()) << phase;
        for (const auto& [identifier, node] : expected.at("nodes").items()) {
            EXPECT_EQ(stored.at("nodes").at(identifier).at("properties"), node.at("properties"))
                << phase << ": " << identifier;
        }

        const FlatSpec flat = flatten(grouped);
        EXPECT_EQ(
            std::set<std::string>(graph->get_identifiers().begin(), graph->get_identifiers().end()),
            flat.nodes)
            << phase;
        std::set<std::string> groups;
        std::ranges::copy(grouped.groups | std::views::keys, std::inserter(groups, groups.end()));
        check_reads(graph, flat.spec, iterations, phase, groups);
    }

    void check_reads(const GraphHandle& graph,
                     const GraphSpec& spec,
                     const uint32_t iterations,
                     const std::string& phase,
                     const std::set<std::string>& skipped = {}) {
        const GraphMemoryReport report = graph->memory_get_report();
        std::ignore = report.to_text();
        aliased_checks += std::ranges::any_of(
            report.transient, [](const auto& output) { return output.placement.aliased; });

        const auto [running, rejected] = running_nodes(spec);
        in_place_errors += rejected.size();

        // one copy per iteration of the period, max delay + 1 without aliasing, one if persistent
        uint32_t period = 1;
        std::map<std::string, uint32_t> max_delay;
        for (const std::string& identifier : running) {
            if (skipped.contains(identifier)) {
                continue;
            }
            for (const auto& [input, read] : connected_reads(spec, running, spec.at(identifier))) {
                period = std::lcm(period, read.delay + 1);
                uint32_t& root_delay =
                    max_delay[fmt::format("{}.{}", *root_of(spec, running, read.src, read.output),
                                          root_output(read.output))];
                root_delay = std::max(root_delay, read.delay);
            }
        }
        EXPECT_EQ(report.period, period) << phase;
        for (const std::string& identifier : running) {
            if (skipped.contains(identifier)) {
                continue;
            }
            const auto copies = [&](const std::string& output) -> uint32_t {
                if (spec.at(identifier).shape.persistent) {
                    return 1;
                }
                const auto delay = max_delay.find(fmt::format("{}.{}", identifier, output));
                return report.aliasing ? period : delay == max_delay.end() ? 1 : delay->second + 1;
            };
            for (const std::string output : {"out", "buf"}) {
                const uint32_t copy_count = copies(output);
                const std::string name = fmt::format("{}.{}", identifier, output);
                const auto transient_copies =
                    std::ranges::count_if(report.transient, [&](const auto& p) {
                        return copy_count == 1 ? p.output == name
                                               : p.output.starts_with(name + " (");
                    });
                EXPECT_EQ(transient_copies, copy_count) << phase << ": " << name;
            }
        }

        for (const auto& [identifier, node_spec] : spec) {
            if (skipped.contains(identifier)) {
                continue;
            }
            const auto node = graph->find_node_for_identifier_and_type<PatternNode>(identifier);
            ASSERT_TRUE(node) << phase << ": " << identifier;
            EXPECT_EQ(node->shape, node_spec.shape) << phase << ": " << identifier;
            EXPECT_EQ(node->process_count, running.contains(identifier) ? iterations : 0)
                << phase << ": " << identifier;
            node->process_count = 0;

            std::map<std::string, Read> inputs;
            if (running.contains(identifier)) {
                inputs = connected_reads(spec, running, node_spec);
            }

            std::map<std::string, uint32_t> record_counts;
            std::map<std::string, uint32_t> checked_counts;
            for (const PatternNode::Record& record : node->records) {
                record_counts[record.input]++;
                const auto input = inputs.find(record.input);
                ASSERT_NE(input, inputs.end())
                    << phase << ": " << identifier << "." << record.input << " is not connected";
                const NodeSpec& src = spec.at(input->second.src);
                passed_on_records += input->second.src != node_spec.inputs.at(record.input).src;
                in_place_records += is_version(input->second.output);
                // a version written in place has the size of the resource it continues
                const NodeSpec& root =
                    spec.at(*root_of(spec, running, input->second.src, input->second.output));
                const uint32_t expected_count =
                    is_image_input(record.input)
                        ? root.shape.width * root.shape.height
                        : std::min(
                              root.shape.buffer_words,
                              static_cast<uint32_t>(PatternNode::SLOT_SIZE / sizeof(uint32_t)));
                ASSERT_EQ(record.word_count, expected_count)
                    << phase << ": " << identifier << "." << record.input;

                checked_counts[record.input]++;
                // a copy read before its producer first wrote it holds its initialization
                const std::string& output = input->second.output;
                const uint32_t expected =
                    record.iteration < input->second.delay
                        ? initial_pattern_value(root.id, root_output(output))
                        : pattern_value(src.id, output,
                                        record.total_iteration - input->second.delay);
                const std::vector<uint32_t> words = node->read(record);
                const auto mismatch = std::find_if(words.begin(), words.end(),
                                                   [&](const uint32_t w) { return w != expected; });
                EXPECT_EQ(mismatch, words.end())
                    << phase << ": " << identifier << "." << record.input << " <- "
                    << input->second.src << " (delay " << input->second.delay << ") iteration "
                    << record.total_iteration << ": word " << (mismatch - words.begin()) << " is "
                    << fmt::format("{:#x}", mismatch == words.end() ? 0 : *mismatch)
                    << ", expected " << fmt::format("{:#x}", expected);
            }
            for (const auto& [input, src] : inputs) {
                EXPECT_EQ(record_counts[input], iterations)
                    << phase << ": " << identifier << "." << input << " <- " << src.src;
                EXPECT_EQ(checked_counts[input], iterations)
                    << phase << ": " << identifier << "." << input << " <- " << src.src;
            }
            node->records.clear();
        }
    }
};
