#include "graph_resources_test.hpp"

#include <ranges>

TEST_F(GraphResourcesTest, SubgraphsReadWhatWasWritten) {
    GroupedSpec grouped = subgraph_spec();
    const GraphHandle graph = create_graph(describe(grouped, true, 2));
    run_and_check(graph, grouped, 6, "loaded");
    for (const uint32_t iterations_in_flight : {1u, 3u, 2u}) {
        set_graph_properties(graph, true, iterations_in_flight);
        run_and_check(graph, grouped, 6, fmt::format("{} in flight", iterations_in_flight));
    }
    const UIProperties props = draw(graph);
    EXPECT_EQ(graph->node_get_state("g"), NodeState::OK);
    EXPECT_TRUE(props.shows_label("Subnodes"));
    EXPECT_TRUE(props.shows_label("[OK] g:c2:c1 (Pattern)"));
    EXPECT_TRUE(props.shows_text("Feeds: [g:c0.inp]", {"g", "inputs", "inp (public)"}));
    EXPECT_TRUE(
        props.shows_text("Receiving from: out, a2 (Pattern)", {"g", "inputs", "inp (public)"}));
    EXPECT_TRUE(props.shows_text("Carried by: g:c2.out", {"g", "outputs", "out (public)"}));
    EXPECT_TRUE(props.shows_text("g:c0.out", {"memory", "blocks"}));

    grouped.nodes.at("g").enabled = false;
    set_node_enabled(graph, "g", false);
    const uint32_t passed_on_before = passed_on_records;
    run_and_check(graph, grouped, 4, "g disabled");
    EXPECT_GT(passed_on_records, passed_on_before);
    const UIProperties disabled = draw(graph);
    EXPECT_EQ(graph->node_get_state("g"), NodeState::BYPASSED);
    EXPECT_EQ(graph->node_get_state("g:c2:c0"), NodeState::FORCE_DISABLED);
    EXPECT_TRUE(disabled.shows_text("g is disabled", {"g:c2:c0"}));

    grouped.nodes.at("g").enabled = true;
    set_node_enabled(graph, "g", true);
    run_and_check(graph, grouped, 4, "g enabled");

    // h fed back into itself: delayed, then undelayed, which the graph removes
    for (const uint32_t delay : {1u, 0u}) {
        GroupSpec& group = grouped.groups.at("h");
        group.children.front().shape.delays[2] = delay;
        graph->node_set_properties("h", group_properties(group));
        if (delay == 1) {
            grouped.nodes.at("h").inputs["in2"] = {"h", 0};
            graph->add_connection("h", "h", "out", "in2");
        } else {
            grouped.nodes.at("h").inputs.erase("in2");
        }
        run_and_check(graph, grouped, 4, fmt::format("h fed back with delay {}", delay));
    }
}

TEST_F(GraphResourcesTest, SubgraphRebuilds) {
    for (uint32_t seed = 0; seed < 4; seed++) {
        std::mt19937 rng(seed);
        GroupedSpec grouped = subgraph_spec();
        GraphHandle graph = create_graph(describe(grouped, true, 2));
        run_and_check(graph, grouped, 3, fmt::format("seed {} loaded", seed));

        uint32_t id = 200;
        const auto random_child = [&](const bool in_place) {
            NodeSpec child{id++, {}, {}};
            PatternShape& shape = child.shape;
            reshape_randomly(rng, shape);
            shape.persistent = random_uint(rng, 0, 3) == 0;
            for (uint32_t i = 0; i < INPUT_COUNT; i++) {
                shape.delays[i] = random_uint(rng, 0, 3) == 0 ? random_uint(rng, 1, 2) : 0;
                shape.required[i] = random_uint(rng, 0, 5) == 0;
            }
            shape.pass_through = shape.delays[0] == 0 && shape.delays[IMAGE_INPUT_COUNT] == 0 &&
                                 random_uint(rng, 0, 3) == 0;
            shape.defaults = random_uint(rng, 0, 1) == 0;
            shape.in_place = in_place || random_uint(rng, 0, 4) == 0;
            return child;
        };
        const auto random_group = [&](const auto& self, const uint32_t depth,
                                      const bool writer) -> GroupSpec {
            GroupSpec group;
            group.settings.prefix = random_uint(rng, 0, 1) == 0 ? "c" : "p";
            group.settings.parallel = random_uint(rng, 0, 2) == 0;
            group.settings.public_in1 = random_uint(rng, 0, 3) == 0;
            const bool with_inner = depth > 0 && random_uint(rng, 0, 2) != 0;
            const bool inner_writes = writer && with_inner && random_uint(rng, 0, 1) == 0;
            const uint32_t count = random_uint(rng, 1, 3);
            for (uint32_t i = 0; i < count; i++) {
                group.children.push_back(random_child(writer && !inner_writes && i == 0));
            }
            if (with_inner) {
                group.inner.push_back(self(self, depth - 1, inner_writes));
            } else {
                // read delayed through the public outputs, which a persistent output would remove
                group.children.back().shape.persistent = false;
            }
            return group;
        };

        for (uint32_t step = 0; step < 12; step++) {
            const std::string phase = fmt::format("seed {} step {}", seed, step);
            switch (random_uint(rng, 0, 5)) {
            case 0:
            case 1:
            case 2: {
                // g always writes in place, r reads its versions
                const std::string identifier = random_uint(rng, 0, 1) == 0 ? "g" : "h";
                const GroupSpec group =
                    random_group(random_group, random_uint(rng, 1, 2), identifier == "g");
                const auto node =
                    graph->find_node_for_identifier_and_type<PatternGroup>(identifier);
                ASSERT_TRUE(node) << phase;
                const std::vector<std::shared_ptr<PatternNode>> kept = node->children;
                // a disabled node is not pre-processed
                if (grouped.nodes.at(identifier).enabled) {
                    node->set_properties(group_properties(group));
                } else {
                    graph->node_set_properties(identifier, group_properties(group));
                }
                grouped.groups.at(identifier) = group;
                run_and_check(graph, grouped, 4, phase + " rebuilt " + identifier);
                for (uint32_t i = 0; i < std::min(kept.size(), node->children.size()); i++) {
                    EXPECT_EQ(node->children[i], kept[i]) << phase;
                    EXPECT_EQ(kept[i]->initialize_count, 1u) << phase;
                    EXPECT_EQ(graph->find_node_for_identifier(
                                  fmt::format("{}:{}{}", identifier, group.settings.prefix, i)),
                              kept[i])
                        << phase;
                }
                continue;
            }
            case 3: {
                const std::string identifier = random_uint(rng, 0, 1) == 0 ? "g" : "h";
                bool& enabled = grouped.nodes.at(identifier).enabled;
                enabled = !enabled;
                set_node_enabled(graph, identifier, enabled);
                run_and_check(graph, grouped, 4, fmt::format("{} {} enabled", phase, identifier));
                continue;
            }
            case 4: {
                const bool alias = random_uint(rng, 0, 1) == 0;
                set_graph_properties(graph, alias, random_uint(rng, 1, 3));
                run_and_check(graph, grouped, 4, phase + " in flight changed");
                continue;
            }
            default: {
                const json stored = graph->to_description().to_json();
                graph = create_graph(stored);
                run_and_check(graph, grouped, 4, phase + " stored and loaded");
            }
            }
        }
    }
}

TEST_F(GraphResourcesTest, AutoConnectionsGiveWayToPublicInputs) {
    GroupedSpec grouped = subgraph_spec();
    grouped.nodes.at("h").inputs["in2"] = {"a", 0};
    const GraphHandle graph = create_graph(describe(grouped, true, 2));
    run_and_check(graph, grouped, 4, "auto connected");
    const auto group = graph->find_node_for_identifier_and_type<PatternGroup>("h");
    ASSERT_TRUE(group);
    PatternGroup::Settings& settings = grouped.groups.at("h").settings;

    for (const bool public_in1 : {true, false}) {
        settings.public_in1 = public_in1;
        group->set_properties(group_properties(grouped.groups.at("h")));
        run_and_check(graph, grouped, 4, fmt::format("public in1 {}", public_in1));
    }

    settings.broken = static_cast<uint32_t>(PatternGroup::Broken::THROWS);
    group->set_properties(group_properties(grouped.groups.at("h")));
    run_and_check(graph, grouped, 4, "failed");
    settings.broken = static_cast<uint32_t>(PatternGroup::Broken::NO);
    settings.public_in1 = true;
    // an erroneous node is not pre-processed
    graph->node_set_properties("h", group_properties(grouped.groups.at("h")));
    run_and_check(graph, grouped, 4, "public in1 after failing");
}

// The child of h reads its own output delayed until h's public in2 is wired.
TEST_F(GraphResourcesTest, UnwiredPublicInputsLeaveAutoConnections) {
    GroupedSpec grouped = subgraph_spec();
    grouped.groups.at("h").settings.public_in1 = true;
    const GraphHandle graph = create_graph(describe(grouped, true, 2));
    run_and_check(graph, grouped, 4, "unwired");

    grouped.nodes.at("h").inputs["in2"] = {"a", 0};
    graph->add_connection("a", "h", "out", "in2");
    run_and_check(graph, grouped, 4, "wired");

    grouped.nodes.at("h").inputs.erase("in2");
    graph->remove_connection("a", "h", "in2");
    run_and_check(graph, grouped, 4, "unwired again");
}

TEST_F(GraphResourcesTest, ChildrenBelongToTheirParent) {
    GroupedSpec grouped = subgraph_spec();
    const GraphHandle graph = create_graph(describe(grouped, true, 2));
    run_and_check(graph, grouped, 2, "loaded");

    EXPECT_THROW(graph->add_connection("a", "g:c0", "out", "in1"), std::invalid_argument);
    EXPECT_THROW(graph->add_connection("g:c0", "a", "out", "in1"), std::invalid_argument);
    EXPECT_FALSE(graph->remove_connection("g:c0", "g:c1", "in0"));
    EXPECT_FALSE(graph->remove_node("g:c0"));
    json described = describe(grouped, true, 2);
    described["nodes"]["x:y"] = {{"type", "Pattern"}};
    EXPECT_THROW(create_graph(described), std::invalid_argument);
    run_and_check(graph, grouped, 2, "rejected edits");

    ASSERT_TRUE(graph->find_node_for_identifier("g:c2:c0"));
    graph->remove_node("g");
    grouped.nodes.erase("g");
    grouped.groups.erase("g");
    for (auto& [identifier, node] : grouped.nodes) {
        std::erase_if(node.inputs, [](const auto& entry) { return entry.second.src == "g"; });
    }
    run_and_check(graph, grouped, 2, "g removed");
}

TEST_F(GraphResourcesTest, UnsupportedAndBrokenSubgraphs) {
    GroupedSpec grouped = subgraph_spec();
    grouped.nodes.at("r").shape.required[0] = true;
    json described = describe(grouped, true, 2);
    described["nodes"]["h"]["type"] = "Unsupported Pattern Group";
    GraphHandle graph = create_graph(described);
    for (uint32_t i = 0; i < 3; i++) {
        graph->run();
    }
    graph->wait();
    EXPECT_EQ(stored_connections(graph), stored_connections(described));
    EXPECT_EQ(graph->node_get_state("h"), NodeState::UNSUPPORTED);
    EXPECT_EQ(graph->node_get_state("r"), NodeState::FORCE_DISABLED);
    EXPECT_FALSE(graph->find_node_for_identifier("h:c0"));

    graph = create_graph(describe(grouped, true, 2));
    run_and_check(graph, grouped, 3, "supported");
    const auto group = graph->find_node_for_identifier_and_type<PatternGroup>("g");
    ASSERT_TRUE(group);
    const auto child = graph->find_node_for_identifier_and_type<PatternNode>("g:c2:c1");
    for (const PatternGroup::Broken broken :
         {PatternGroup::Broken::MISSING_INPUT, PatternGroup::Broken::UNKNOWN_NODE,
          PatternGroup::Broken::THROWS}) {
        const std::string phase = fmt::format("broken {}", static_cast<uint32_t>(broken));
        GroupSpec& spec = grouped.groups.at("g");
        spec.settings.broken = static_cast<uint32_t>(broken);
        group->set_properties(group_properties(spec));
        run_and_check(graph, grouped, 3, phase);
        const UIProperties props = draw(graph);
        EXPECT_EQ(graph->node_get_state("g"), NodeState::ERRORED) << phase;
        EXPECT_EQ(graph->node_get_state("g:c2:c1"), NodeState::FORCE_DISABLED) << phase;
        EXPECT_TRUE(props.shows_text("g has an error", {"g:c2:c1"})) << phase;
        EXPECT_EQ(graph->find_node_for_identifier("g:c2:c1"), child) << phase;

        spec.settings.broken = static_cast<uint32_t>(PatternGroup::Broken::NO);
        // an erroneous node is not pre-processed
        graph->node_set_properties("g", group_properties(spec));
        run_and_check(graph, grouped, 3, phase + " fixed");
    }
}

TEST_F(GraphResourcesTest, ParentsWithPortsOfTheirOwn) {
    GroupedSpec grouped = subgraph_spec();
    grouped.groups.at("g").settings.own_input = "in2";
    grouped.groups.at("h").settings.own_input = "own";
    grouped.nodes.at("h").inputs["own"] = {"h", 0, "out"};
    const GraphHandle graph = create_graph(describe(grouped, true, 2));
    run_and_check(graph, grouped, 3, "own inputs");
    const UIProperties props = draw(graph);
    EXPECT_TRUE(props.shows_text("Receiving from: out, a (Pattern)", {"g", "inputs", "in2"}));
    EXPECT_FALSE(props.shows_text("Optional input not connected.", {"g", "inputs", "in2"}));
    EXPECT_TRUE(props.shows_text("Receiving from: out, h:c0 (Pattern)", {"h", "inputs", "own"}));
    EXPECT_EQ(graph->node_get_state("h"), NodeState::OK);

    const NodeConnectors connectors = graph->node_get_connectors("g");
    const auto names = connectors.inputs | std::views::transform(&NodeConnectors::Input::name);
    ASSERT_FALSE(names.empty());
    EXPECT_EQ(names.front(), "in2");
    EXPECT_TRUE(std::ranges::is_sorted(names | std::views::drop(1)));
    EXPECT_NE(std::ranges::find(names, "inp"), names.end());
    EXPECT_NE(std::ranges::find(connectors.outputs, "out", &NodeConnectors::Output::name),
              connectors.outputs.end());
    EXPECT_EQ(connectors.connections.at("in2"), (std::pair<std::string, std::string>{"a", "out"}));
    EXPECT_EQ(connectors.connections.at("inp"), (std::pair<std::string, std::string>{"a2", "out"}));

    grouped.groups.at("g").settings.own_output = true;
    graph->node_set_properties("g", {{"own output", true}});
    run_and_check(graph, grouped, 3, "own output");
    const UIProperties own_output = draw(graph);
    EXPECT_EQ(graph->node_get_state("g"), NodeState::ERRORED);
    EXPECT_TRUE(own_output.shows_text("declares output out, which its subgraph provides", {"g"}));
    EXPECT_EQ(graph->node_get_state("g:c0"), NodeState::FORCE_DISABLED);
    EXPECT_EQ(graph->node_get_connectors("g").connections.at("in2").first, "a");

    grouped.groups.at("g").settings.own_output = false;
    graph->node_set_properties("g", {{"own output", false}});
    run_and_check(graph, grouped, 3, "own output removed");
}

// g's required delayed input is the persistent z through y, found only after its children connect.
TEST_F(GraphResourcesTest, ParentsStoppingLateStopTheirChildren) {
    GroupedSpec grouped = subgraph_spec();
    PatternGroup::Settings& settings = grouped.groups.at("g").settings;
    settings.own_input = "own";
    settings.own_delay = 1;
    settings.own_required = true;
    NodeSpec& persistent = add_node(grouped.nodes, "z");
    persistent.shape.persistent = true;
    persistent.inputs["in0"] = {"g", 0};
    NodeSpec& bypassed = add_node(grouped.nodes, "y");
    bypassed.enabled = false;
    bypassed.shape.pass_through = true;
    bypassed.inputs["in0"] = {"z", 0};
    grouped.nodes.at("g").inputs["own"] = {"y", 1};
    const json described = describe(grouped, true, 2);
    const GraphHandle graph = create_graph(described);
    for (uint32_t i = 0; i < 3; i++) {
        graph->run();
    }
    graph->wait();
    forget_runs(graph);
    for (uint32_t i = 0; i < 2; i++) {
        graph->run();
    }
    graph->wait();

    EXPECT_EQ(stored_connections(graph), stored_connections(described));
    const UIProperties props = draw(graph);
    EXPECT_EQ(graph->node_get_state("g"), NodeState::ERRORED);
    EXPECT_TRUE(props.shows_text("output is persistent", {"g"}));
    EXPECT_EQ(graph->node_get_state("y"), NodeState::BYPASSED);
    EXPECT_EQ(graph->node_get_state("z"), NodeState::OK);
    for (const std::string identifier : {"g:c0", "g:c1", "g:c2:c0", "g:c2:c1"}) {
        EXPECT_EQ(graph->node_get_state(identifier), NodeState::FORCE_DISABLED) << identifier;
        EXPECT_EQ(graph->find_node_for_identifier_and_type<PatternNode>(identifier)->process_count,
                  0u)
            << identifier;
    }
}

TEST_F(GraphResourcesTest, SubgraphsSwappingNames) {
    const GraphHandle graph =
        create_graph(json{{"version", 4},
                          {"nodes",
                           {{"s", {{"type", "Swap Group"}, {"outputs", {"out->r.in0"}}}},
                            {"src", {{"type", "Pattern"}, {"outputs", {"out->s.in0"}}}},
                            {"r", {{"type", "Pattern"}}}}}});
    const auto group = graph->find_node_for_identifier_and_type<SwapGroup>("s");
    ASSERT_TRUE(group);
    const std::array<std::shared_ptr<PatternGroup>, 2>& groups = group->groups;
    const auto run = [&] {
        forget_runs(graph);
        for (uint32_t i = 0; i < 3; i++) {
            graph->run();
        }
        graph->wait();
    };
    for (const bool swapped : {false, true, false, true}) {
        graph->node_set_properties("s", {{"swapped", swapped}});
        run();
        const std::array<std::string, 2> names =
            swapped ? std::array<std::string, 2>{"b", "a"} : std::array<std::string, 2>{"a", "b"};
        for (uint32_t i = 0; i < 2; i++) {
            EXPECT_EQ(graph->find_node_for_identifier("s:" + names[i]), groups[i]) << swapped;
            for (uint32_t c = 0; c < groups[i]->children.size(); c++) {
                const std::string identifier = fmt::format("s:{}:c{}", names[i], c);
                EXPECT_EQ(graph->find_node_for_identifier(identifier), groups[i]->children[c])
                    << identifier << " " << swapped;
                EXPECT_GT(groups[i]->children[c]->process_count, 0u)
                    << identifier << " " << swapped;
                EXPECT_EQ(groups[i]->children[c]->initialize_count, 1u) << identifier;
            }
        }
        EXPECT_EQ(std::ranges::distance(graph->get_identifiers()), swapped ? 8 : 9) << swapped;
        ASSERT_FALSE(graph->find_node_for_identifier_and_type<PatternNode>("r")->records.empty());
    }

    // the first group fails while renamed: its children, also the one it lost, follow it and stop
    graph->node_set_properties("s", {{"swapped", false}});
    run();
    const std::vector<std::shared_ptr<PatternNode>> first_children = groups[0]->children;
    graph->node_set_properties("s", {{"swapped", true}, {"fail", true}});
    run();
    EXPECT_EQ(graph->node_get_state("s:b"), NodeState::ERRORED);
    EXPECT_EQ(std::ranges::distance(graph->get_identifiers()), 9);
    for (uint32_t c = 0; c < 2; c++) {
        const std::string identifier = fmt::format("s:b:c{}", c);
        EXPECT_EQ(graph->find_node_for_identifier(identifier), first_children[c]) << identifier;
        EXPECT_EQ(first_children[c]->process_count, 0u) << identifier;
        EXPECT_EQ(graph->find_node_for_identifier(fmt::format("s:a:c{}", c)),
                  groups[1]->children[c])
            << identifier;
        EXPECT_GT(groups[1]->children[c]->process_count, 0u) << identifier;
    }
    graph->node_set_properties("s", {{"fail", false}});
    run();
    EXPECT_EQ(std::ranges::distance(graph->get_identifiers()), 8);
    EXPECT_FALSE(graph->find_node_for_identifier("s:b:c1"));
    EXPECT_GT(first_children[0]->process_count, 0u);
}

TEST_F(GraphResourcesTest, InvalidSubgraphsError) {
    const GraphHandle graph =
        create_graph(json{{"version", 4},
                          {"nodes",
                           {{"s", {{"type", "Scripted Group"}, {"outputs", {"out->r.in0"}}}},
                            {"src", {{"type", "Pattern"}, {"outputs", {"out->s.in"}}}},
                            {"r", {{"type", "Pattern"}}}}}});
    const auto group = graph->find_node_for_identifier_and_type<ScriptedGroup>("s");
    ASSERT_TRUE(group);
    const auto valid = [](ScriptedGroup& scripted) {
        Subgraph subgraph;
        subgraph.add_node("c0", scripted.children[0]);
        subgraph.add_node("c1", scripted.children[1]);
        subgraph.add_connection(".in", "c0.in0");
        subgraph.add_connection("c0.out", "c1.in0");
        subgraph.add_connection("c1.out", ".out");
        subgraph.add_pass_through(".in", ".out");
        return subgraph;
    };
    const auto run = [&] {
        graph->restart_run();
        forget_runs(graph);
        for (uint32_t i = 0; i < 3; i++) {
            graph->run();
        }
        graph->wait();
    };
    const std::set<std::string> wiring = {"src.out->s.in", "s.out->r.in0"};
    group->script = valid;
    run();
    ASSERT_EQ(graph->node_get_state("s"), NodeState::OK);

    const NodeHandle unregistered = std::make_shared<UnregisteredNode>();
    const std::vector<std::pair<std::string, std::function<void(ScriptedGroup&, Subgraph&)>>>
        cases = {
            {"reserved", [](ScriptedGroup& s, Subgraph& g) { g.add_node("a:b", s.children[0]); }},
            {"cannot be empty",
             [](ScriptedGroup& s, Subgraph& g) { g.add_node("", s.children[0]); }},
            {"is taken", [](ScriptedGroup& s, Subgraph& g) { g.add_node("c0", s.children[1]); }},
            {"null, its parent or added twice",
             [](ScriptedGroup&, Subgraph& g) { g.add_node("x", nullptr); }},
            {"null, its parent or added twice",
             [](ScriptedGroup& s, Subgraph& g) { g.add_node("x", s.children[0]); }},
            {"null, its parent or added twice",
             [](ScriptedGroup& s, Subgraph& g) { g.add_node("x", s.shared_from_this()); }},
            {"was not registered",
             [&](ScriptedGroup&, Subgraph& g) { g.add_node("x", unregistered); }},
            {"already in the graph as src",
             [&](ScriptedGroup&, Subgraph& g) {
                 g.add_node("x", graph->find_node_for_identifier("src"));
             }},
            {"connects two public connectors",
             [](ScriptedGroup&, Subgraph& g) { g.add_connection(".in", ".other"); }},
            {"connected twice",
             [](ScriptedGroup&, Subgraph& g) { g.add_connection(".in", "c1.in0"); }},
            {"connected twice",
             [](ScriptedGroup&, Subgraph& g) { g.add_connection("c0.out", ".out"); }},
            {"unknown connector",
             [](ScriptedGroup&, Subgraph& g) { g.add_connection("x.out", ".out"); }},
            {"unknown connector",
             [](ScriptedGroup&, Subgraph& g) { g.add_connection("c0", ".out"); }},
            {"invalid pass-through",
             [](ScriptedGroup&, Subgraph& g) { g.add_pass_through(".in", ".other"); }},
            {"invalid pass-through",
             [](ScriptedGroup&, Subgraph& g) {
                 g.add_connection("c0.buf", ".buf");
                 g.add_pass_through(".unconnected", ".buf");
             }},
            {"input missing of s:c1, which does not exist",
             [](ScriptedGroup&, Subgraph& g) { g.add_connection(".unconnected", "c1.missing"); }},
            {"input missing of s:c1, which does not exist",
             [](ScriptedGroup&, Subgraph& g) { g.add_connection(".in", "c1.missing"); }},
            {"two inputs pass through to out",
             [](ScriptedGroup& s, Subgraph&) {
                 s.inputs = {{.name = "own",
                              .connector = VkImageIn::create(),
                              .optional = true,
                              .pass_through = "out"}};
             }},
        };
    for (const auto& [error, breaking] : cases) {
        group->script = [&](ScriptedGroup& scripted) {
            Subgraph subgraph = valid(scripted);
            breaking(scripted, subgraph);
            return subgraph;
        };
        run();
        const UIProperties props = draw(graph);
        EXPECT_EQ(graph->node_get_state("s"), NodeState::ERRORED) << error;
        EXPECT_TRUE(props.shows_text(error, {"s"})) << error;
        EXPECT_EQ(stored_connections(graph), wiring) << error;
        for (uint32_t c = 0; c < 2; c++) {
            EXPECT_EQ(graph->find_node_for_identifier(fmt::format("s:c{}", c)), group->children[c])
                << error;
            EXPECT_EQ(group->children[c]->process_count, 0u) << error;
        }

        group->inputs.clear();
        group->script = valid;
        run();
        EXPECT_EQ(graph->node_get_state("s"), NodeState::OK) << error;
        EXPECT_GT(group->children[1]->process_count, 0u) << error;
    }

    // a public output of a missing child output errors the subgraph, the wiring stays
    group->script = [](ScriptedGroup& scripted) {
        Subgraph subgraph;
        subgraph.add_node("c0", scripted.children[0]);
        subgraph.add_node("c1", scripted.children[1]);
        subgraph.add_connection(".in", "c0.in0");
        subgraph.add_connection("c0.out", "c1.in0");
        subgraph.add_connection("c1.missing", ".out");
        return subgraph;
    };
    run();
    EXPECT_EQ(stored_connections(graph), wiring);
    EXPECT_TRUE(draw(graph).shows_text("Optional input not connected.", {"r", "inputs", "in0"}));
    group->script = valid;
    run();

    const NodeHandle throwing = std::make_shared<ThrowingNode>();
    group->script = [&](ScriptedGroup& scripted) {
        Subgraph subgraph = valid(scripted);
        subgraph.add_node("t", throwing);
        return subgraph;
    };
    run();
    EXPECT_EQ(graph->node_get_state("s:t"), NodeState::UNSUPPORTED);
    EXPECT_EQ(graph->node_get_state("s"), NodeState::OK);
    group->script = valid;
    run();

    graph->add_connection("src", "s", "buf", "in");
    run();
    EXPECT_EQ(stored_connections(graph), std::set<std::string>({"s.out->r.in0"}));

    graph->add_connection("src", "s", "out", "in");
    run();
    EXPECT_EQ(stored_connections(graph), wiring);
    JSONLoadProperties props(
        json{{"nodes", {{"s", {{"inputs", {{"in (public)", {{"Remove Connection", true}}}}}}}}}});
    graph->draw_ui(props);
    run();
    EXPECT_EQ(stored_connections(graph), std::set<std::string>({"s.out->r.in0"}));
}

TEST_F(GraphResourcesTest, AutoConnectionsThroughPublicInputsKeepTheWiring) {
    GroupedSpec grouped;
    NodeSpec& persistent = add_node(grouped.nodes, "p");
    persistent.shape.persistent = true;
    add_node(grouped.nodes, "h").inputs = {{"in0", {"p", 0}}, {"bin0", {"p", 0}}};
    add_node(grouped.nodes, "r").inputs = {{"in0", {"h", 0}}, {"bin0", {"h", 0}}};
    PatternShape with_defaults{.defaults = true};
    with_defaults.delays[4] = 1;
    grouped.groups["h"] = {.children = {{100, with_defaults, {}}}};
    const GraphHandle graph = create_graph(describe(grouped, true, 2));
    run_and_check(graph, grouped, 4, "persistent source");
    EXPECT_TRUE(
        draw(graph).shows_text("Optional input not connected.", {"h:c0", "inputs", "bin1"}));
}

TEST_F(GraphResourcesTest, ChildrenMovingBetweenParents) {
    const GraphHandle graph =
        create_graph(json{{"version", 4},
                          {"nodes",
                           {{"m", {{"type", "Move Group"}, {"outputs", {"out->r.in0"}}}},
                            {"src", {{"type", "Pattern"}, {"outputs", {"out->m.in0"}}}},
                            {"r", {{"type", "Pattern"}}}}}});
    const auto group = graph->find_node_for_identifier_and_type<MoveGroup>("m");
    ASSERT_TRUE(group);
    using Place = MoveGroup::Place;
    const std::map<Place, std::string> identifiers = {
        {Place::A, "m:a:c1"}, {Place::B, "m:b:c1"}, {Place::HERE, "m:x"}};
    for (const Place place :
         {Place::A, Place::B, Place::A, Place::HERE, Place::B, Place::HERE, Place::A}) {
        const std::string phase = identifiers.at(place);
        graph->node_set_properties("m", {{"place", static_cast<uint32_t>(place)}});
        forget_runs(graph);
        for (uint32_t i = 0; i < 3; i++) {
            graph->run();
        }
        graph->wait();
        EXPECT_EQ(graph->node_get_state("m"), NodeState::OK) << phase;
        EXPECT_EQ(graph->find_node_for_identifier(phase), group->x) << phase;
        EXPECT_EQ(group->x->initialize_count, 1u) << phase;
        EXPECT_GT(group->x->process_count, 0u) << phase;
        // m, a, b, their first children, x, src, r
        EXPECT_EQ(std::ranges::distance(graph->get_identifiers()), 8) << phase;
    }
}

// A child moving to a later parent leaves its name to a new child of its old parent.
TEST_F(GraphResourcesTest, ChildrenMovingLeaveTheirName) {
    const GraphHandle graph =
        create_graph(json{{"version", 4},
                          {"nodes",
                           {{"m", {{"type", "Move Group"}, {"outputs", {"out->r.in0"}}}},
                            {"src", {{"type", "Pattern"}, {"outputs", {"out->m.in0"}}}},
                            {"r", {{"type", "Pattern"}}}}}});
    const auto group = graph->find_node_for_identifier_and_type<MoveGroup>("m");
    ASSERT_TRUE(group);
    graph->node_set_properties("m", {{"replace", true}});
    using Place = MoveGroup::Place;
    for (const Place place : {Place::A, Place::B, Place::A, Place::B}) {
        const std::string phase = place == Place::A ? "in a" : "in b";
        graph->node_set_properties("m", {{"place", static_cast<uint32_t>(place)}});
        run(graph);
        EXPECT_EQ(graph->find_node_for_identifier("m:a:c1"),
                  place == Place::A ? group->x : group->stand_ins[0])
            << phase;
        EXPECT_EQ(graph->find_node_for_identifier("m:b:c1"),
                  place == Place::B ? group->x : group->stand_ins[1])
            << phase;
        // m, a, b, their two children, src, r
        EXPECT_EQ(std::ranges::distance(graph->get_identifiers()), 9) << phase;
        EXPECT_EQ(group->x->initialize_count, 1u) << phase;
    }
}

TEST_F(GraphResourcesTest, DisabledParentsPassVersionsOn) {
    const GraphHandle graph = create_graph(
        json{{"version", 4},
             {"nodes",
              {{"s", {{"type", "Scripted Group"}, {"enabled", false}, {"outputs", {"out->r.in0"}}}},
               {"src", {{"type", "Pattern"}, {"outputs", {"out->s.a"}}}},
               {"r", {{"type", "Pattern"}}}}}});
    const auto group = graph->find_node_for_identifier_and_type<ScriptedGroup>("s");
    ASSERT_TRUE(group);
    group->children[0]->shape.in_place = true;
    group->script = [](ScriptedGroup& scripted) {
        Subgraph subgraph;
        subgraph.add_node("c0", scripted.children[0]);
        subgraph.add_node("c1", scripted.children[1]);
        subgraph.add_connection(".a", "c0.inp");
        subgraph.add_connection("c0.inp", ".out");
        subgraph.add_connection(".b", "c1.in0");
        subgraph.add_pass_through(".b", ".out");
        return subgraph;
    };
    graph->restart_run();
    for (uint32_t i = 0; i < 3; i++) {
        graph->run();
    }
    graph->wait();
    const auto reader = graph->find_node_for_identifier_and_type<PatternNode>("r");
    EXPECT_EQ(reader->process_count, 3u);
    EXPECT_EQ(std::ranges::count(reader->records, std::string("in0"), &PatternNode::Record::input),
              3);
    EXPECT_TRUE(
        draw(graph).shows_text("Receiving from: out, src (Pattern)", {"r", "inputs", "in0"}));
}

TEST_F(GraphResourcesTest, ParentsPassThrough) {
    const GraphHandle graph = create_graph(
        json{{"version", 4},
             {"nodes",
              {{"s", {{"type", "Scripted Group"}, {"enabled", false}, {"outputs", {"out->r.in0"}}}},
               {"src", {{"type", "Pattern"}, {"outputs", {"out->s.in"}}}},
               {"r", {{"type", "Pattern"}}}}}});
    const auto group = graph->find_node_for_identifier_and_type<ScriptedGroup>("s");
    ASSERT_TRUE(group);
    for (const bool own : {false, true}) {
        group->inputs.clear();
        if (own) {
            group->inputs.push_back({.name = "in",
                                     .connector = VkImageIn::create(),
                                     .access = ConnectorAccess::transfer_src,
                                     .optional = true,
                                     .pass_through = "out"});
        }
        group->script = [own](ScriptedGroup& scripted) {
            Subgraph subgraph;
            subgraph.add_node("c0", scripted.children[0]);
            subgraph.add_connection(".in", "c0.in0");
            subgraph.add_connection("c0.out", ".out");
            if (!own) {
                subgraph.add_pass_through(".in", ".out");
            }
            return subgraph;
        };
        graph->restart_run();
        forget_runs(graph);
        for (uint32_t i = 0; i < 3; i++) {
            graph->run();
        }
        graph->wait();
        const UIProperties props = draw(graph);
        EXPECT_EQ(graph->node_get_state("s"), NodeState::BYPASSED) << own;
        EXPECT_TRUE(props.shows_text("Receiving from: out, src (Pattern)", {"r", "inputs", "in0"}))
            << own;
        EXPECT_EQ(group->children[0]->process_count, 0u) << own;
        EXPECT_EQ(graph->find_node_for_identifier_and_type<PatternNode>("r")->process_count, 3u)
            << own;
    }
}
