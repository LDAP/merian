#include "graph_resources_test.hpp"

TEST_F(GraphResourcesTest, ChainReadsWhatWasWritten) {
    GraphSpec spec;
    for (uint32_t index = 0; index < 6; index++) {
        spec[node_name(index)] = {index + 1, {}, {}};
        spec[node_name(index)].shape.scratch = index % 2 == 0;
        if (index > 0) {
            spec[node_name(index)].inputs["in0"] = {node_name(index - 1), 0};
            spec[node_name(index)].inputs["bin0"] = {node_name(index - 1), 0};
        }
    }
    const GraphHandle graph = create_graph();
    graph->load_from_json(describe(spec, true, 2));
    run_and_check(graph, spec, 6, "chain");
    EXPECT_LT(graph->get_memory_report().blocks.at(0).size,
              graph->get_memory_report().blocks.at(0).requested_size);
}

TEST_F(GraphResourcesTest, LayersReadWhatWasWrittenWhileProfiled) {
    GraphSpec spec;
    add_node(spec, "a");
    for (const std::string identifier : {"b", "c", "d"}) {
        NodeSpec& node = add_node(spec, identifier);
        node.inputs["in0"] = {"a", 0};
        node.inputs["bin0"] = {"a", 0};
    }
    NodeSpec& e = add_node(spec, "e");
    e.inputs["in0"] = {"b", 0};
    e.inputs["in1"] = {"c", 0};
    e.inputs["in2"] = {"d", 0};

    const GraphHandle graph = load(spec);
    run_and_check(graph, spec, 6, "overlapping");
    graph->set_profiler_enabled(true);
    run_and_check(graph, spec, 6, "profiled");
#ifdef MERIAN_PROFILER_ENABLE
    const Profiler::Report report = graph->get_run_report();
    const auto run_nodes =
        std::ranges::find(report.gpu_report, "Run nodes", &Profiler::ReportEntry::name);
    ASSERT_NE(run_nodes, report.gpu_report.end());
    EXPECT_EQ(run_nodes->children.size(), spec.size());
#endif
}

TEST_F(GraphResourcesTest, RandomGraphsReadWhatWasWritten) {
    std::mt19937 rng(1234);
    for (uint32_t test = 0; test < 40; test++) {
        const uint32_t node_count = random_uint(rng, 2, 14);
        GraphSpec spec = random_spec(rng, node_count);
        const std::string name = fmt::format("graph {} ({} nodes)", test, node_count);
        SCOPED_TRACE(name);

        const GraphHandle graph = create_graph();
        graph->load_from_json(describe(spec, true, random_uint(rng, 1, 3)));
        run_and_check(graph, spec, 7, "load");

        set_graph_properties(graph, false, 2);
        run_and_check(graph, spec, 5, "aliasing off");

        set_graph_properties(graph, true, random_uint(rng, 1, 3));
        run_and_check(graph, spec, 5, "iterations in flight");

        const std::set<std::string> running = running_nodes(spec).running;
        for (auto& [identifier, node] : spec) {
            if (running.contains(identifier) && random_uint(rng, 0, 1) == 0) {
                reshape_randomly(rng, node.shape);
                graph->find_node_for_identifier_and_type<PatternNode>(identifier)
                    ->set_shape(node.shape);
            }
        }
        run_and_check(graph, spec, 6, "reshape");

        const uint32_t rewired = random_uint(rng, 0, node_count - 1);
        const NodeSpec before = spec.at(node_name(rewired));
        connect_randomly(rng, spec, rewired);
        const NodeSpec& after = spec.at(node_name(rewired));
        // the delays must change together with the connections, not one run later
        graph->find_node_for_identifier_and_type<PatternNode>(node_name(rewired))->shape =
            after.shape;
        graph->request_reconnect();
        for (const auto& [input, src] : before.inputs) {
            graph->remove_connection(src.src, node_name(rewired), input);
        }
        for (const auto& [input, src] : after.inputs) {
            graph->add_connection(src.src, node_name(rewired), source_output(input, src), input);
        }
        run_and_check(graph, spec, 6, "rewire");

        const std::string removed = node_name(random_uint(rng, 0, node_count - 1));
        graph->remove_node(removed);
        spec.erase(removed);
        for (auto& [identifier, node] : spec) {
            std::erase_if(node.inputs,
                          [&](const auto& input) { return input.second.src == removed; });
        }
        run_and_check(graph, spec, 6, "remove");

        std::vector<std::string> bypassed;
        for (auto& [identifier, node] : spec) {
            if (random_uint(rng, 0, 2) == 0) {
                bypassed.push_back(identifier);
                set_node_enabled(graph, identifier, false);
                node.enabled = false;
            }
        }
        run_and_check(graph, spec, 6, "bypass");
        for (const std::string& identifier : bypassed) {
            set_node_enabled(graph, identifier, true);
            spec.at(identifier).enabled = true;
        }
        run_and_check(graph, spec, 6, "unbypass");

        const std::string disabled =
            std::next(spec.begin(), random_uint(rng, 0, spec.size() - 1))->first;
        set_node_enabled(graph, disabled, false);
        spec.at(disabled).enabled = false;
        run_and_check(graph, spec, 6, "disable");

        const json stored = graph->store_to_json();
        const GraphHandle restored = create_graph();
        restored->load_from_json(stored);
        run_and_check(restored, spec, 6, "restore");
        EXPECT_EQ(restored->store_to_json().at("nodes"), stored.at("nodes"));
        EXPECT_EQ(restored->store_to_json().at("graph_properties"), stored.at("graph_properties"));

        set_node_enabled(restored, disabled, true);
        spec.at(disabled).enabled = true;
        run_and_check(restored, spec, 6, "enable");
    }
    EXPECT_GT(aliased_checks, 0);
    EXPECT_GT(passed_on_records, 0);
    EXPECT_GT(in_place_records, 0);
    EXPECT_GT(in_place_errors, 0);
}

TEST_F(GraphResourcesTest, RuntimeErrorsAndRejectionsSettle) {
    GraphSpec spec;
    add_node(spec, "a");
    NodeSpec& w1 = add_node(spec, "w1");
    w1.shape.in_place = true;
    w1.inputs["inp"] = {"a", 0, "out"};
    NodeSpec& w2 = add_node(spec, "w2");
    w2.shape.in_place = true;
    w2.inputs["inp"] = {"a", 0, "out"};
    NodeSpec& n = add_node(spec, "n");
    n.shape.fail_connected = true;
    n.inputs["in0"] = {"a", 0};

    const GraphHandle graph = load(spec);
    run_and_check(graph, spec, 4, "settle");
    graph->request_reconnect();
    run_and_check(graph, spec, 4, "settle again");
    EXPECT_EQ(running_nodes(spec).rejected, (std::set<std::string>{"w1", "w2"}));

    // b writes in place only while a, which fails, does not run: both stay out
    GraphSpec alternating;
    add_node(alternating, "s");
    NodeSpec& a = add_node(alternating, "a");
    a.shape.fail_connected = true;
    a.inputs["in0"] = {"s", 0};
    a.inputs["in1"] = {"b", 0};
    NodeSpec& b = add_node(alternating, "b");
    b.shape.in_place = true;
    b.shape.fail_connected = true;
    b.inputs["inp"] = {"s", 0, "out"};
    run_and_check(load(alternating), alternating, 4, "alternating");
    EXPECT_EQ(running_nodes(alternating).running, (std::set<std::string>{"s"}));

    // x requests a reconnect whenever its consumer, which fails, appears or disappears
    GraphSpec reconnecting;
    add_node(reconnecting, "x").shape.reconnect_on_change = true;
    NodeSpec& y = add_node(reconnecting, "y");
    y.shape.fail_connected = true;
    y.inputs["in0"] = {"x", 0};
    run_and_check(load(reconnecting), reconnecting, 4, "reconnecting");
    EXPECT_EQ(running_nodes(reconnecting).running, (std::set<std::string>{"x"}));

    // w fails while processing: its reader receives the version w received from then on
    GraphSpec failing;
    add_node(failing, "s");
    NodeSpec& w = add_node(failing, "w");
    w.shape.in_place = true;
    w.shape.fail_process = true;
    w.inputs["inp"] = {"s", 0, "out"};
    add_node(failing, "r").inputs["in0"] = {"w", 0, "inp"};
    const GraphHandle failing_graph = load(failing);
    failing_graph->run();
    failing_graph->wait();
    forget_runs(failing_graph);
    run_and_check(failing_graph, failing, 4, "failed while processing");
    EXPECT_EQ(failing_graph->get_node_state("w"), NodeState::ERRORED);
}
