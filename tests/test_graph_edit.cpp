#include "graph_test.hpp"

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"

#include <filesystem>

using namespace merian;
using json = nlohmann::json;

namespace {

class EventNode : public Node, public TimeProvider {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{.name = "src",
                 .connector = con_src,
                 .access = ConnectorAccess::compute_read,
                 .optional = true}};
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& io_layout) override {
        if (!listen.empty()) {
            io_layout.register_event_listener(
                listen, [this](const GraphEvent::Info& info, const GraphEvent::Data&) {
                    received.push_back(
                        fmt::format("{}/{}/{}", info.node_name, info.identifier, info.event_name));
                });
        }
        return {{.name = "out", .connector = con_out, .access = ConnectorAccess::compute_write}};
    }

    NodeStatusFlags on_connected(const NodeIOLayout& /*io_layout*/,
                                 const NodeIO& /*io*/,
                                 const NodeConnectionInfo& /*info*/,
                                 Submission& /*submission*/) override {
        return RESET_IN_FLIGHT_DATA | removal("on_connected");
    }

    NodeStatusFlags pre_process(const NodeIO& /*io*/, const NodeProcessInfo& /*info*/) override {
        return removal("pre_process");
    }

    std::optional<std::chrono::nanoseconds> provide_time() override {
        return std::nullopt;
    }

    NodeStatusFlags process(const NodeIO& io,
                            const NodeProcessInfo& /*info*/,
                            Submission& /*submission*/) override {
        io.send_event("tick");
        return removal("process");
    }

    NodeStatusFlags properties(Properties& props) override {
        NodeStatusFlags flags = props.config_text("listen", listen) ? NEEDS_RECONNECT : 0;
        bool remove = false;
        if (props.config_bool("remove", remove) && remove) {
            flags |= REMOVE_NODE;
        }
        std::ignore = props.config_text("remove in", remove_in);
        return flags;
    }

    std::string listen;
    std::string remove_in;
    std::vector<std::string> received;

  private:
    NodeStatusFlags removal(const std::string& callback) const {
        return remove_in == callback ? NodeStatusFlags{REMOVE_NODE} : NodeStatusFlags{};
    }

    const VkSampledImageInHandle con_src = VkSampledImageIn::create();
    const ManagedVkImageOutHandle con_out =
        ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 8, 8);
};

class EventGroup : public Node {
  public:
    EventGroup() {
        child->listen = "//ping";
    }

    Subgraph describe_subgraph() override {
        Subgraph subgraph;
        subgraph.add_node("c", child);
        subgraph.add_connection("c.out", ".out");
        return subgraph;
    }

    const std::shared_ptr<EventNode> child = std::make_shared<EventNode>();
};

class FailingInitializeNode : public Node {
  public:
    void initialize(const ContextHandle& /*context*/,
                    const ResourceAllocatorHandle& /*allocator*/) override {
        throw std::runtime_error{"initialize fails"};
    }
};

class UILoadProperties : public JSONLoadProperties {
  public:
    using JSONLoadProperties::JSONLoadProperties;

    bool is_ui() override {
        return true;
    }
};

class GraphEditTest : public GraphTest {
  protected:
    static void SetUpTestSuite() {
        NodeRegistry& registry = NodeRegistry::get_instance();
        registry.register_node_type<EventNode>("Event Test", "test-graph-edit");
        registry.register_node_type<EventGroup>("Event Group Test", "test-graph-edit");
        registry.register_node_type<FailingInitializeNode>("Failing Initialize Test",
                                                           "test-graph-edit");
        create_context("test-graph-edit");
    }

    using GraphTest::create_graph;

    static GraphHandle create_graph(const json& description) {
        const GraphHandle graph = GraphTest::create_graph(description);
        run(graph);
        return graph;
    }

    static json chain() {
        return json{{"version", 4},
                    {"nodes",
                     {{"a", {{"type", "Event Test"}, {"outputs", {"out->b.src", "out->c.src"}}}},
                      {"b", {{"type", "Event Test"}}},
                      {"c", {{"type", "Event Test"}}}}}};
    }
};

} // namespace

// The Remove button, REMOVE_NODE from Node::properties, and both in one draw.
TEST_F(GraphEditTest, RemovalWhileDrawingTheUi) {
    for (const json& node_props : {json{{"Remove", true}}, json{{"properties", {{"remove", true}}}},
                                   json{{"Remove", true}, {"properties", {{"remove", true}}}}}) {
        SCOPED_TRACE(node_props.dump());
        const GraphHandle graph = create_graph(chain());

        JSONLoadProperties props(json{{"nodes", {{"a", node_props}}}});
        graph->draw_ui(props);

        EXPECT_FALSE(graph->find_node_for_identifier("a"));
        EXPECT_TRUE(stored_connections(graph).empty());
        std::ignore = draw(graph);
        run(graph);
        EXPECT_EQ(graph->node_get_state("b"), NodeState::OK);
    }
}

TEST_F(GraphEditTest, RemovalWhileRunning) {
    for (const std::string callback : {"on_connected", "pre_process", "process"}) {
        const GraphHandle graph = create_graph(chain());

        graph->node_set_properties("b", {{"remove in", callback}});
        graph->restart_run();
        run(graph);

        EXPECT_FALSE(graph->find_node_for_identifier("b")) << callback;
        EXPECT_EQ(stored_connections(graph), std::set<std::string>{"a.out->c.src"}) << callback;
        EXPECT_EQ(graph->node_get_state("c"), NodeState::OK) << callback;
    }
}

TEST_F(GraphEditTest, RemovalWhileLoading) {
    json description = chain();
    description["nodes"]["b"]["properties"] = {{"remove", true}};
    const GraphHandle graph = create_graph(description);

    EXPECT_FALSE(graph->find_node_for_identifier("b"));
    EXPECT_EQ(stored_connections(graph), std::set<std::string>{"a.out->c.src"});
    EXPECT_EQ(graph->node_get_state("c"), NodeState::OK);
}

TEST_F(GraphEditTest, RemovedNodesReceiveNoEvents) {
    json description = chain();
    description["nodes"]["b"]["properties"] = {{"listen", "//ping,/graph/connect"}};
    const GraphHandle graph = create_graph(description);
    const auto removed = graph->find_node_for_identifier_and_type<EventNode>("b");
    removed->received.clear();

    ASSERT_TRUE(graph->remove_node("b"));
    graph->send_event("ping");
    run(graph);

    EXPECT_TRUE(removed->received.empty());
}

TEST_F(GraphEditTest, InvalidEditsThrowOrFail) {
    const GraphHandle graph = create_graph(chain());

    EXPECT_THROW(graph->add_connection("x", "b", "out", "src"), std::invalid_argument);
    EXPECT_THROW(graph->remove_connection("x", "b", "src"), std::invalid_argument);
    EXPECT_THROW(std::ignore = graph->node_get_state("x"), std::invalid_argument);
    EXPECT_FALSE(graph->remove_node("x"));
    EXPECT_FALSE(graph->remove_connection("b", "c", "src"));
    EXPECT_TRUE(has_connection(graph, "a", "out->c.src"));
    for (const std::string identifier : {"", "user", "graph", "a:b", "a.b", "a/b", "a->b", "a"}) {
        EXPECT_THROW(graph->add_node("Event Test", identifier), std::invalid_argument)
            << identifier;
    }
}

TEST_F(GraphEditTest, FailingInitializeIsUnsupported) {
    const GraphHandle graph = create_graph();
    graph->add_node("Failing Initialize Test", "f");
    run(graph);

    EXPECT_EQ(graph->node_get_state("f"), NodeState::UNSUPPORTED);
}

TEST_F(GraphEditTest, LoadIntoARunningGraph) {
    const GraphHandle graph = create_graph(chain());
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "test-graph-edit-load.json";
    create_graph(json{{"version", 4}, {"nodes", {{"x", {{"type", "Event Test"}}}}}})
        ->to_description()
        .to_file(path);

    UILoadProperties props(json{{"path", path.string()}, {"Load", true}});
    graph->draw_ui(props);
    run(graph);
    std::filesystem::remove(path);

    EXPECT_EQ(
        std::set<std::string>(graph->get_identifiers().begin(), graph->get_identifiers().end()),
        std::set<std::string>{"x"});
    EXPECT_EQ(graph->node_get_state("x"), NodeState::OK);
}

TEST_F(GraphEditTest, EventsReachChildren) {
    const GraphHandle graph = create_graph(json{
        {"version", 4},
        {"nodes",
         {{"g", {{"type", "Event Group Test"}}},
          {"l", {{"type", "Event Test"}, {"properties", {{"listen", "Event Test/g:c/tick"}}}}}}}});
    run(graph);
    graph->send_event("ping");

    const auto listener = graph->find_node_for_identifier_and_type<EventNode>("l");
    ASSERT_FALSE(listener->received.empty());
    EXPECT_TRUE(std::ranges::all_of(listener->received, [](const std::string& event) {
        return event == "Event Test/g:c/tick";
    }));
    const auto group = graph->find_node_for_identifier_and_type<EventGroup>("g");
    EXPECT_NE(std::ranges::find(group->child->received, "/user/ping"),
              group->child->received.end());
}
