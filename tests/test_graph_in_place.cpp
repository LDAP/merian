#include "graph_resources_test.hpp"

TEST_F(GraphResourcesTest, InPlaceVersions) {
    GraphSpec spec;
    add_node(spec, "a").shape.width = 32;
    spec.at("a").shape.height = 24;
    NodeSpec& w1 = add_node(spec, "w1");
    w1.shape.in_place = true;
    w1.inputs["inp"] = {"a", 0, "out"};
    NodeSpec& w2 = add_node(spec, "w2");
    w2.shape.in_place = true;
    w2.inputs["inp"] = {"w1", 0, "inp"};
    // sort after the writers of what they read, which share their layer unless ordered
    add_node(spec, "y0").inputs["in0"] = {"a", 0};
    add_node(spec, "y1").inputs["in0"] = {"w1", 0, "inp"};
    NodeSpec& y2 = add_node(spec, "y2");
    y2.inputs["in0"] = {"w2", 0, "inp"};
    y2.inputs["in1"] = {"w2", 1, "inp"};
    y2.shape.delays[1] = 1;

    const GraphHandle graph = load(spec);
    run_and_check(graph, spec, 6, "in place");
    EXPECT_TRUE(draw(graph).shows_label("inp (in place)"));

    set_graph_properties(graph, false, 3);
    run_and_check(graph, spec, 6, "in place, aliasing off");
    set_graph_properties(graph, true, 2);

    set_node_enabled(graph, "w1", false);
    spec.at("w1").enabled = false;
    run_and_check(graph, spec, 6, "in place, w1 disabled");
    EXPECT_EQ(graph->get_node_state("w1"), NodeState::BYPASSED);
    set_node_enabled(graph, "w1", true);
    spec.at("w1").enabled = true;
    run_and_check(graph, spec, 6, "in place, w1 enabled");

    const auto set_required = [&](const bool required) {
        spec.at("w1").shape.required[2] = required;
        graph->find_node_for_identifier_and_type<PatternNode>("w1")->shape = spec.at("w1").shape;
        graph->request_reconnect();
    };
    set_required(true);
    run_and_check(graph, spec, 6, "in place, w1 errors");
    set_required(false);
    run_and_check(graph, spec, 6, "in place, w1 recovers");

    const json stored = graph->store_to_json();
    const GraphHandle restored = create_graph();
    restored->load_from_json(stored);
    run_and_check(restored, spec, 6, "in place, restored");
    EXPECT_EQ(restored->store_to_json().at("nodes"), stored.at("nodes"));

    restored->remove_node("w1");
    spec.erase("w1");
    spec.at("w2").inputs.erase("inp");
    spec.at("y1").inputs.erase("in0");
    run_and_check(restored, spec, 6, "in place, w1 removed");
}

TEST_F(GraphResourcesTest, ChainsUseTheResourceOfTheirRoot) {
    GraphSpec spec;
    add_node(spec, "a").shape.counting = true;
    NodeSpec& w1 = add_node(spec, "w1");
    w1.shape.in_place = true;
    w1.inputs["inp"] = {"a", 0, "out"};
    NodeSpec& w2 = add_node(spec, "w2");
    w2.shape.in_place = true;
    w2.inputs["inp"] = {"w1", 0, "inp"};
    NodeSpec& y = add_node(spec, "y");
    y.shape.sampled = true;
    y.inputs["in0"] = {"w2", 0, "inp"};

    const GraphHandle graph = load(spec);
    constexpr uint32_t iterations = 5;
    run_and_check(graph, spec, iterations, "chain");

    const CountingImageOut& out =
        *graph->find_node_for_identifier_and_type<PatternNode>("a")->counting;
    EXPECT_EQ(out.created, 1);
    EXPECT_EQ(out.connected, 1);
    EXPECT_EQ(out.pre_processed, iterations);
    EXPECT_EQ(out.post_processed, iterations);
    EXPECT_TRUE(out.access.image_usage & vk::ImageUsageFlagBits::eSampled);

    const GraphMemoryReport report = graph->get_memory_report();
    const auto root = std::ranges::find_if(
        report.placed, [](const auto& output) { return output.output.starts_with("a.out"); });
    ASSERT_NE(root, report.placed.end());
    EXPECT_TRUE(root->lifetime.contains(report.layer_count - 1));
    EXPECT_FALSE(std::ranges::any_of(
        report.placed, [](const auto& output) { return output.output.ends_with(".inp"); }));
}

TEST_F(GraphResourcesTest, WritersThatStopLatePassOn) {
    GraphSpec spec;
    add_node(spec, "a");
    NodeSpec& w1 = add_node(spec, "w1");
    w1.shape.in_place = true;
    w1.shape.fail_outputs = true;
    w1.inputs["inp"] = {"a", 0, "out"};
    NodeSpec& r1 = add_node(spec, "r1");
    r1.inputs["in0"] = {"w1", 0, "inp"};
    r1.inputs["in1"] = {"w1", 1, "inp"};
    r1.shape.delays[1] = 1;
    r1.shape.required[0] = true;

    // w2 misses its required delayed input from the disabled d
    add_node(spec, "d").enabled = false;
    NodeSpec& w2 = add_node(spec, "w2");
    w2.shape.in_place = true;
    w2.inputs["binp"] = {"a", 0, "buf"};
    w2.inputs["in1"] = {"d", 1};
    w2.shape.delays[1] = 1;
    w2.shape.required[1] = true;
    NodeSpec& r2 = add_node(spec, "r2");
    r2.inputs["bin0"] = {"w2", 0, "binp"};
    r2.shape.required[IMAGE_INPUT_COUNT] = true;

    // w3 errors after connecting r3: its required delayed input is the persistent x through p
    add_node(spec, "b");
    NodeSpec& w3 = add_node(spec, "w3");
    w3.shape.in_place = true;
    w3.inputs["inp"] = {"b", 0, "out"};
    w3.inputs["in1"] = {"p", 1};
    w3.shape.delays[1] = 1;
    w3.shape.required[1] = true;
    NodeSpec& x = add_node(spec, "x");
    x.shape.persistent = true;
    x.inputs["in0"] = {"w3", 0};
    NodeSpec& p = add_node(spec, "p");
    p.shape.pass_through = true;
    p.enabled = false;
    p.inputs["in0"] = {"x", 0};
    NodeSpec& r3 = add_node(spec, "r3");
    r3.inputs["in0"] = {"w3", 0, "inp"};
    r3.shape.required[0] = true;
    NodeSpec& f3 = add_node(spec, "f3");
    f3.inputs["in0"] = {"w3", 0};
    f3.shape.required[0] = true;

    // w4 waits on a cycle with y; d4 reads its version delayed, r4 without delay
    add_node(spec, "c");
    NodeSpec& w4 = add_node(spec, "w4");
    w4.shape.in_place = true;
    w4.inputs["inp"] = {"c", 0, "out"};
    w4.inputs["in0"] = {"y", 0};
    add_node(spec, "y").inputs["in0"] = {"w4", 0};
    NodeSpec& d4 = add_node(spec, "d4");
    d4.inputs["in0"] = {"w4", 1, "inp"};
    d4.shape.delays[0] = 1;
    d4.shape.required[0] = true;
    NodeSpec& r4 = add_node(spec, "r4");
    r4.inputs["in0"] = {"w4", 0, "inp"};
    r4.shape.required[0] = true;

    // w5 fails to describe its outputs, w6 after it misses a required delayed input
    add_node(spec, "e");
    NodeSpec& w5 = add_node(spec, "w5");
    w5.shape.in_place = true;
    w5.shape.fail_outputs = true;
    w5.inputs["inp"] = {"e", 0, "out"};
    NodeSpec& w6 = add_node(spec, "w6");
    w6.shape.in_place = true;
    w6.inputs["inp"] = {"w5", 0, "inp"};
    w6.inputs["in1"] = {"d", 1};
    w6.shape.delays[1] = 1;
    w6.shape.required[1] = true;
    NodeSpec& r6 = add_node(spec, "r6");
    r6.inputs["in0"] = {"w6", 0, "inp"};
    r6.shape.required[0] = true;

    // the disabled w7 passes r7's own output back to it
    NodeSpec& r7 = add_node(spec, "r7");
    r7.inputs["in0"] = {"w7", 0, "inp"};
    NodeSpec& w7 = add_node(spec, "w7");
    w7.shape.in_place = true;
    w7.enabled = false;
    w7.inputs["inp"] = {"r7", 0, "out"};

    const GraphHandle graph = load(spec);
    run_and_check(graph, spec, 6, "late");
    const std::set<std::string> running = running_nodes(spec).running;
    for (const std::string node : {"r1", "r2", "r3", "x", "d4", "r4", "r6", "r7"}) {
        EXPECT_TRUE(running.contains(node)) << node;
    }
    for (const std::string node : {"w3", "w4", "y", "w6", "f3"}) {
        EXPECT_FALSE(running.contains(node)) << node;
    }
    EXPECT_EQ(graph->get_node_state("f3"), NodeState::FORCE_DISABLED);
}

TEST_F(GraphResourcesTest, WritersOfPlainSourcesThatStopLateStop) {
    GraphSpec spec;
    add_node(spec, "d").enabled = false;
    NodeSpec& a = add_node(spec, "a");
    a.shape.delays[0] = 1;
    a.shape.required[0] = true;
    a.inputs["in0"] = {"d", 1};
    NodeSpec& w = add_node(spec, "w");
    w.shape.in_place = true;
    w.shape.required_in_place = true;
    w.inputs["inp"] = {"a", 0, "out"};
    w.inputs["binp"] = {"a", 0, "buf"};

    const GraphHandle graph = load(spec);
    run_and_check(graph, spec, 4, "stopped late");
    EXPECT_EQ(graph->get_node_state("a"), NodeState::FORCE_DISABLED);
    EXPECT_EQ(graph->get_node_state("w"), NodeState::FORCE_DISABLED);
}

TEST_F(GraphResourcesTest, DelayedReadsOfVersionsStayWired) {
    GraphSpec spec;
    add_node(spec, "p").shape.persistent = true;
    add_node(spec, "q");
    NodeSpec& w = add_node(spec, "w");
    w.shape.in_place = true;
    w.inputs["inp"] = {"p", 0, "out"};
    NodeSpec& r = add_node(spec, "r");
    r.inputs["in0"] = {"w", 1, "inp"};
    r.shape.delays[0] = 1;

    const GraphHandle graph = load(spec);
    run_and_check(graph, spec, 6, "persistent root");

    graph->remove_connection("p", "w", "inp");
    graph->add_connection("q", "w", "out", "inp");
    spec.at("w").inputs["inp"] = {"q", 0, "out"};
    run_and_check(graph, spec, 6, "rewired root");
    EXPECT_TRUE(connected_reads(spec, running_nodes(spec).running, spec.at("r")).contains("in0"));
}

TEST_F(GraphResourcesTest, InvalidInPlaceWritesError) {
    const auto check = [&](const GraphSpec& spec, const std::set<std::string>& errors,
                           const std::string& phase) {
        const GraphHandle graph = load(spec);
        run_and_check(graph, spec, 4, phase);
        EXPECT_EQ(running_nodes(spec).rejected, errors) << phase;
        for (const auto& [identifier, node] : spec) {
            EXPECT_EQ(graph->get_node_state(identifier),
                      errors.contains(identifier) ? NodeState::ERRORED : NodeState::OK)
                << phase << ": " << identifier;
        }
    };
    const auto writer = [](GraphSpec& spec, const std::string& identifier,
                           const InputSpec& source) {
        NodeSpec& node = add_node(spec, identifier);
        node.shape.in_place = true;
        node.inputs["inp"] = source;
    };

    GraphSpec two_writers;
    add_node(two_writers, "a");
    writer(two_writers, "w1", {"a", 0, "out"});
    writer(two_writers, "w2", {"a", 0, "out"});
    check(two_writers, {"w1", "w2"}, "two writers");

    GraphSpec delayed;
    add_node(delayed, "a");
    writer(delayed, "w1", {"a", 0, "out"});
    NodeSpec& r = add_node(delayed, "r");
    r.inputs["in0"] = {"a", 1};
    r.shape.delays[0] = 1;
    check(delayed, {"r"}, "delayed reader");

    GraphSpec self_read;
    add_node(self_read, "a");
    writer(self_read, "w1", {"a", 0, "out"});
    self_read.at("w1").inputs["in0"] = {"a", 0};
    check(self_read, {"w1"}, "writer reads its version");

    // r reads the version before and after w1: w1 cannot run after it, w2 after r can
    GraphSpec cycle;
    add_node(cycle, "a");
    writer(cycle, "w1", {"a", 0, "out"});
    NodeSpec& cycle_reader = add_node(cycle, "r");
    cycle_reader.inputs["in0"] = {"a", 0};
    cycle_reader.inputs["in1"] = {"w1", 0, "inp"};
    writer(cycle, "w2", {"r", 0, "out"});
    check(cycle, {"w1"}, "cycle");

    // without the cyclic w1, nobody writes what d reads delayed
    GraphSpec cycle_first = cycle;
    NodeSpec& d = add_node(cycle_first, "d");
    d.inputs["in0"] = {"a", 1};
    d.shape.delays[0] = 1;
    check(cycle_first, {"w1"}, "cycle before delayed reader");

    // the cyclic w1 goes first, so w3 is the only writer
    GraphSpec cycle_before_writers = cycle;
    writer(cycle_before_writers, "w3", {"a", 0, "out"});
    check(cycle_before_writers, {"w1"}, "cycle before two writers");

    // without the two writers, nobody writes what d reads delayed
    GraphSpec writers_first = two_writers;
    NodeSpec& writers_reader = add_node(writers_first, "d");
    writers_reader.inputs["in0"] = {"a", 1};
    writers_reader.shape.delays[0] = 1;
    check(writers_first, {"w1", "w2"}, "two writers before delayed reader");
}

TEST_F(GraphResourcesTest, PointersInPlace) {
    const GraphHandle graph = create_graph();
    graph->load_from_json(json::parse(R"({
        "version": 4,
        "nodes": {
            "s": {"type": "Int Source", "outputs": ["out->w.inp", "out->r0.in"]},
            "w": {"type": "Int Add", "outputs": ["inp->r1.in"]},
            "r0": {"type": "Int Reader"},
            "r1": {"type": "Int Reader"}
        }
    })"));
    constexpr uint32_t iterations = 4;
    for (uint32_t i = 0; i < iterations; i++) {
        graph->run();
    }
    graph->wait();
    const auto values = [&](const std::string& identifier) {
        return graph->find_node_for_identifier_and_type<IntReader>(identifier)->values;
    };
    std::vector<int> expected(iterations);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(values("r0"), expected);
    std::ranges::transform(expected, expected.begin(),
                           [](const int value) { return value + 1000; });
    EXPECT_EQ(values("r1"), expected);
}

TEST_F(GraphResourcesTest, SourcesThatCannotBeWrittenInPlace) {
    const GraphHandle graph = create_graph();
    graph->load_from_json(json::parse(R"({
        "version": 4,
        "nodes": {
            "s": {"type": "Fixed Int Source", "outputs": ["out->w.inp"]},
            "w": {"type": "Int Add", "outputs": ["inp->r.in"]},
            "r": {"type": "Int Reader"}
        }
    })"));
    constexpr uint32_t iterations = 3;
    for (uint32_t i = 0; i < iterations; i++) {
        graph->run();
    }
    graph->wait();
    EXPECT_EQ(graph->get_node_state("w"), NodeState::ERRORED);
    std::vector<int> expected(iterations);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(graph->find_node_for_identifier_and_type<IntReader>("r")->values, expected);
}

TEST_F(GraphResourcesTest, CanvasesWrittenInPlace) {
    const GraphHandle graph = create_graph();
    graph->load_from_json(json::parse(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Pattern", "properties": {"id": 7}, "outputs": ["out->blit.src"]},
            "canvas": {"type": "Canvas Source", "outputs": ["canvas->blit.canvas"]},
            "blit": {"type": "Swapchain Blit", "outputs": ["canvas->r.in0"]},
            "r": {"type": "Pattern", "properties": {"id": 8}}
        }
    })"));
    const auto check = [&](const uint32_t iterations, const bool gaps) {
        graph->find_node_for_identifier_and_type<CanvasSource>("canvas")->gaps = gaps;
        graph->request_reconnect();
        for (uint32_t i = 0; i < iterations; i++) {
            graph->run();
        }
        graph->wait();
        const auto reader = graph->find_node_for_identifier_and_type<PatternNode>("r");
        EXPECT_EQ(reader->records.size(), gaps ? (iterations + 1) / 2 : iterations);
        for (const PatternNode::Record& record : reader->records) {
            EXPECT_FALSE(gaps && record.iteration % 2 == 1);
            const uint32_t expected = pattern_value(7, "out", record.total_iteration);
            for (const uint32_t word : reader->read(record)) {
                ASSERT_EQ(word, expected) << "iteration " << record.total_iteration;
            }
        }
        reader->records.clear();
    };
    check(4, false);
    check(5, true);
    EXPECT_EQ(graph->get_node_state("blit"), NodeState::OK);
}
