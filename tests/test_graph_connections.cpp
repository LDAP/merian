#include "graph_test.hpp"

#include "merian-graph/connectors/buffer/vk_buffer_out_managed.hpp"
#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"

using namespace merian;
using json = nlohmann::json;

namespace {

class FeedbackNode : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {
            {.name = "src",
             .connector = con_src,
             .access = ConnectorAccess::compute_read,
             .optional = true,
             .pass_through = "out"},
            {.name = "prev",
             .connector = con_prev,
             .access = ConnectorAccess::compute_read,
             .delay = 1,
             .optional = true,
             .auto_connect = AutoConnect::output("out")},
            {.name = "hist",
             .connector = con_hist,
             .access = ConnectorAccess::compute_read,
             .delay = 1,
             .optional = true,
             .auto_connect = AutoConnect::input("src")},
        };
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {{"out", ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 64, 64),
                 ConnectorAccess::compute_write}};
    }

  private:
    const VkSampledImageInHandle con_src = VkSampledImageIn::create();
    const VkSampledImageInHandle con_prev = VkSampledImageIn::create();
    const VkSampledImageInHandle con_hist = VkSampledImageIn::create();
};

class UnsupportedNode : public FeedbackNode {
  public:
    void initialize(const ContextHandle& /*context*/,
                    const ResourceAllocatorHandle& /*allocator*/) override {
        throw graph_errors::node_error{"not on this device"};
    }
};

class RequiredNode : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{.name = "src", .connector = con_src, .access = ConnectorAccess::compute_read}};
    }

  private:
    const VkSampledImageInHandle con_src = VkSampledImageIn::create();
};

// Its output cannot be read delayed.
class PersistentNode : public Node {
  public:
    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {{.name = "out",
                 .connector = ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 64, 64),
                 .access = ConnectorAccess::compute_write,
                 .persistent = true}};
    }
};

class MissingFeedbackNode : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{.name = "prev",
                 .connector = con_prev,
                 .access = ConnectorAccess::compute_read,
                 .delay = 1,
                 .auto_connect = AutoConnect::output("missing")}};
    }

  private:
    const VkSampledImageInHandle con_prev = VkSampledImageIn::create();
};

class DeclarationNode : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        std::vector<InputConnectorDescriptor> inputs = {
            {.name = "canvas",
             .connector = con_canvas,
             .access = ConnectorAccess::compute_read_write,
             .optional = true,
             .in_place = true}};
        if (mode == "delayed") {
            inputs[0].delay = 1;
        } else if (mode == "passing") {
            inputs[0].pass_through = "out";
        } else if (mode == "passed onto") {
            inputs.push_back(
                {.name = "src", .connector = con_src, .optional = true, .pass_through = "canvas"});
        } else if (mode == "named twice") {
            inputs.push_back({.name = "canvas", .connector = con_src, .optional = true});
        } else if (mode == "history of it") {
            inputs.push_back({.name = "src",
                              .connector = con_src,
                              .access = ConnectorAccess::compute_read,
                              .delay = 1,
                              .optional = true,
                              .auto_connect = AutoConnect::input("canvas")});
        } else if (mode == "auto connection cycle") {
            inputs.push_back({.name = "src",
                              .connector = con_src,
                              .access = ConnectorAccess::compute_read,
                              .delay = 1,
                              .optional = true,
                              .auto_connect = AutoConnect::input("hist")});
            inputs.push_back({.name = "hist",
                              .connector = con_hist,
                              .access = ConnectorAccess::compute_read,
                              .delay = 1,
                              .optional = true,
                              .auto_connect = AutoConnect::input("src")});
        } else if (mode == "read only") {
            inputs[0].access = ConnectorAccess::compute_read;
        } else if (mode == "same resource twice") {
            inputs.push_back({.name = "src",
                              .connector = con_src,
                              .access = ConnectorAccess::compute_read_write,
                              .optional = true,
                              .in_place = true});
        }
        return inputs;
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        if (mode == "connector error") {
            throw graph_errors::connector_error{"declared wrongly"};
        }
        return {{mode == "output named like it" ? "canvas" : "out",
                 ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 64, 64),
                 ConnectorAccess::compute_write}};
    }

    NodeStatusFlags properties(Properties& props) override {
        return props.config_text("mode", mode) ? NEEDS_RECONNECT : 0;
    }

  private:
    std::string mode;
    const VkSampledImageInHandle con_canvas = VkSampledImageIn::create();
    const VkSampledImageInHandle con_src = VkSampledImageIn::create();
    const VkSampledImageInHandle con_hist = VkSampledImageIn::create();
};

class AutoConnectChainNode : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {
            {.name = "later",
             .connector = con_later,
             .access = ConnectorAccess::compute_read,
             .delay = 2,
             .optional = true,
             .auto_connect = AutoConnect::input("earlier")},
            {.name = "earlier",
             .connector = con_earlier,
             .access = ConnectorAccess::compute_read,
             .delay = 1,
             .optional = true,
             .auto_connect = AutoConnect::input("src")},
            {.name = "src",
             .connector = con_src,
             .access = ConnectorAccess::compute_read,
             .optional = true},
        };
    }

  private:
    const VkSampledImageInHandle con_later = VkSampledImageIn::create();
    const VkSampledImageInHandle con_earlier = VkSampledImageIn::create();
    const VkSampledImageInHandle con_src = VkSampledImageIn::create();
};

class UnsupportedWriterNode : public Node {
  public:
    void initialize(const ContextHandle& /*context*/,
                    const ResourceAllocatorHandle& /*allocator*/) override {
        throw graph_errors::node_error{"not on this device"};
    }

    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{.name = "canvas",
                 .connector = con_canvas,
                 .access = ConnectorAccess::compute_read_write,
                 .in_place = true}};
    }

  private:
    const VkSampledImageInHandle con_canvas = VkSampledImageIn::create();
};

class BufferNode : public Node {
  public:
    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {{.name = "buf",
                 .connector = ManagedVkBufferOut::create(
                     vk::BufferCreateInfo{{}, 1024, vk::BufferUsageFlagBits::eStorageBuffer}),
                 .access = ConnectorAccess::compute_write}};
    }
};

class GraphConnectionsTest : public GraphTest {
  protected:
    static void SetUpTestSuite() {
        NodeRegistry& registry = NodeRegistry::get_instance();
        registry.register_node_type<FeedbackNode>("Feedback Test", "test-graph-connections");
        registry.register_node_type<UnsupportedNode>("Unsupported Test", "test-graph-connections");
        registry.register_node_type<RequiredNode>("Required Test", "test-graph-connections");
        registry.register_node_type<PersistentNode>("Persistent Test", "test-graph-connections");
        registry.register_node_type<MissingFeedbackNode>("Missing Feedback Test",
                                                         "test-graph-connections");
        registry.register_node_type<BufferNode>("Buffer Test", "test-graph-connections");
        registry.register_node_type<DeclarationNode>("Declaration Test", "test-graph-connections");
        registry.register_node_type<UnsupportedWriterNode>("Unsupported Writer Test",
                                                           "test-graph-connections");
        registry.register_node_type<AutoConnectChainNode>("Auto Connect Chain Test",
                                                          "test-graph-connections");
        create_context("test-graph-connections");
    }

    using GraphTest::create_graph;

    static GraphHandle create_graph(const std::string& description) {
        const GraphHandle graph = create_graph();
        graph->load_from_json(json::parse(description));
        run(graph);
        return graph;
    }
};

} // namespace

TEST_F(GraphConnectionsTest, AutoConnectionsAreConnected) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->f.src"]},
            "f": {"type": "Feedback Test"}
        }
    })");

    EXPECT_TRUE(has_connection(graph, "a", "out->a.prev"));
    EXPECT_TRUE(has_connection(graph, "f", "out->f.prev"));
    EXPECT_TRUE(has_connection(graph, "a", "out->f.hist"));
    // a's src is not connected
    EXPECT_FALSE(has_connection(graph, "a", "out->a.hist"));
}

TEST_F(GraphConnectionsTest, ChainedAutoConnectionsAreConnected) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->n.src"]},
            "n": {"type": "Auto Connect Chain Test"}
        }
    })");

    EXPECT_TRUE(has_connection(graph, "a", "out->n.earlier"));
    EXPECT_TRUE(has_connection(graph, "a", "out->n.later"));
}

TEST_F(GraphConnectionsTest, ChainedAutoConnectionsFollowRewiredInput) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->n.src"]},
            "c": {"type": "Feedback Test"},
            "n": {"type": "Auto Connect Chain Test"}
        }
    })");
    ASSERT_TRUE(has_connection(graph, "a", "out->n.later"));

    graph->add_connection("c", "n", "out", "src");
    run(graph);
    EXPECT_TRUE(has_connection(graph, "c", "out->n.earlier"));
    EXPECT_TRUE(has_connection(graph, "c", "out->n.later"));
}

TEST_F(GraphConnectionsTest, ExplicitWiringWins) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->f.src", "out->f.prev"]},
            "f": {"type": "Feedback Test"}
        }
    })");

    EXPECT_TRUE(has_connection(graph, "a", "out->f.prev"));
    EXPECT_FALSE(has_connection(graph, "f", "out->f.prev"));
}

TEST_F(GraphConnectionsTest, RemovedAutoConnectionStaysRemoved) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {"f": {"type": "Feedback Test"}}
    })");
    ASSERT_TRUE(has_connection(graph, "f", "out->f.prev"));

    graph->remove_connection("f", "f", "prev");
    run(graph);
    graph->request_reconnect();
    run(graph);
    EXPECT_FALSE(has_connection(graph, "f", "out->f.prev"));

    const GraphHandle restored = create_graph();
    restored->load_from_json(graph->store_to_json());
    run(restored);
    EXPECT_TRUE(has_connection(restored, "f", "out->f.prev"));
}

TEST_F(GraphConnectionsTest, AutoConnectionWithoutOutputErrors) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {"m": {"type": "Missing Feedback Test"}}
    })");
    graph->request_reconnect();
    run(graph);

    EXPECT_FALSE(has_connection(graph, "m", "missing->m.prev"));
    EXPECT_EQ(graph->get_node_state("m"), NodeState::ERRORED);
}

TEST_F(GraphConnectionsTest, HistoryFollowsItsInputOnceConnected) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {"a": {"type": "Feedback Test"}, "f": {"type": "Feedback Test"}}
    })");
    EXPECT_FALSE(has_connection(graph, "a", "out->f.hist"));

    graph->add_connection("a", "f", "out", "src");
    run(graph);
    EXPECT_TRUE(has_connection(graph, "a", "out->f.hist"));
}

TEST_F(GraphConnectionsTest, HistoryFollowsRewiredInput) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->f.src"]},
            "c": {"type": "Feedback Test"},
            "d": {"type": "Feedback Test"},
            "f": {"type": "Feedback Test"}
        }
    })");
    ASSERT_TRUE(has_connection(graph, "a", "out->f.hist"));

    graph->add_connection("c", "f", "out", "src");
    run(graph);
    EXPECT_TRUE(has_connection(graph, "c", "out->f.hist"));

    graph->add_connection("d", "f", "out", "hist");
    graph->add_connection("a", "f", "out", "src");
    run(graph);
    EXPECT_TRUE(has_connection(graph, "d", "out->f.hist"));
}

TEST_F(GraphConnectionsTest, HistoryFollowsRemovedAndConnectedInput) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->f.src"]},
            "c": {"type": "Feedback Test"},
            "d": {"type": "Feedback Test"},
            "f": {"type": "Feedback Test"}
        }
    })");
    ASSERT_TRUE(has_connection(graph, "a", "out->f.hist"));

    graph->remove_connection("a", "f", "src");
    run(graph);
    EXPECT_FALSE(has_connection(graph, "a", "out->f.hist"));
    graph->add_connection("c", "f", "out", "src");
    run(graph);
    EXPECT_TRUE(has_connection(graph, "c", "out->f.hist"));

    // unless it was rewired itself
    graph->add_connection("d", "f", "out", "hist");
    graph->remove_connection("c", "f", "src");
    graph->add_connection("a", "f", "out", "src");
    run(graph);
    EXPECT_TRUE(has_connection(graph, "d", "out->f.hist"));
}

TEST_F(GraphConnectionsTest, RejectedHistoryFollowsNewSource) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "p": {"type": "Persistent Test", "outputs": ["out->f.src"]},
            "c": {"type": "Feedback Test"},
            "d": {"type": "Feedback Test"},
            "f": {"type": "Feedback Test"}
        }
    })");
    EXPECT_TRUE(has_connection(graph, "p", "out->f.src"));
    EXPECT_FALSE(has_connection(graph, "p", "out->f.hist"));
    EXPECT_EQ(graph->get_node_state("f"), NodeState::OK);

    graph->add_connection("c", "f", "out", "src");
    run(graph);
    EXPECT_TRUE(has_connection(graph, "c", "out->f.hist"));

    // a removal by the user holds
    graph->remove_connection("c", "f", "hist");
    graph->add_connection("d", "f", "out", "src");
    run(graph);
    EXPECT_FALSE(has_connection(graph, "d", "out->f.hist"));
}

TEST_F(GraphConnectionsTest, HistoryReconnectsAfterSourceRemoval) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->f.src"]},
            "c": {"type": "Feedback Test"},
            "f": {"type": "Feedback Test"}
        }
    })");
    graph->remove_node("a");
    run(graph);

    graph->add_connection("c", "f", "out", "src");
    run(graph);
    EXPECT_TRUE(has_connection(graph, "c", "out->f.hist"));
}

TEST_F(GraphConnectionsTest, DisabledNodesPassThrough) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->f.src"]},
            "f": {"type": "Feedback Test", "enabled": false, "outputs": ["out->g.src"]},
            "g": {"type": "Feedback Test", "enabled": false, "outputs": ["out->b.src"]},
            "b": {"type": "Feedback Test"}
        }
    })");

    const UIProperties props = draw(graph);
    EXPECT_EQ(graph->get_node_state("f"), NodeState::BYPASSED);
    EXPECT_EQ(graph->get_node_state("g"), NodeState::BYPASSED);
    EXPECT_EQ(graph->get_node_state("b"), NodeState::OK);
    EXPECT_TRUE(props.shows_text("Receiving from: out, a (Feedback Test)", {"b", "src"}));
    EXPECT_TRUE(has_connection(graph, "g", "out->b.src"));
}

TEST_F(GraphConnectionsTest, UnsupportedNodesDoNotPassThrough) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->u.src"]},
            "u": {"type": "Unsupported Test", "outputs": ["out->r.src"]},
            "r": {"type": "Required Test"}
        }
    })");

    EXPECT_EQ(graph->get_node_state("u"), NodeState::UNSUPPORTED);
    EXPECT_EQ(graph->get_node_state("r"), NodeState::FORCE_DISABLED);
}

TEST_F(GraphConnectionsTest, UnsupportedWritersPassTheirVersionOn) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["out->u.canvas"]},
            "u": {"type": "Unsupported Writer Test", "outputs": ["canvas->r.src"]},
            "r": {"type": "Required Test"}
        }
    })");

    EXPECT_EQ(graph->get_node_state("u"), NodeState::UNSUPPORTED);
    EXPECT_EQ(graph->get_node_state("r"), NodeState::OK);
}

TEST_F(GraphConnectionsTest, InPlaceDeclarationsError) {
    for (const std::string mode : {"delayed", "read only", "passing", "passed onto", "named twice",
                                   "history of it", "auto connection cycle", "same resource twice",
                                   "output named like it", "connector error"}) {
        const GraphHandle graph = create_graph(fmt::format(R"({{
            "version": 4,
            "nodes": {{
                "a": {{"type": "Feedback Test", "outputs": ["out->d.canvas", "out->d.src"]}},
                "d": {{"type": "Declaration Test", "properties": {{"mode": "{}"}}}}
            }}
        }})",
                                                           mode));
        const UIProperties props = draw(graph);
        EXPECT_EQ(graph->get_node_state("d"), NodeState::ERRORED) << mode;
        if (mode == std::string{"history of it"}) {
            EXPECT_TRUE(props.shows_text("invalid auto connection")) << mode;
        }
        if (mode == std::string{"read only"}) {
            EXPECT_TRUE(props.shows_text("written in place but declares only reads")) << mode;
        }
        if (mode == std::string{"auto connection cycle"}) {
            EXPECT_TRUE(props.shows_text("auto connected to itself")) << mode;
        }
    }
}

TEST_F(GraphConnectionsTest, WiringOfDisabledNodesIsChecked) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Feedback Test", "outputs": ["missing->f.hist"]},
            "f": {"type": "Feedback Test", "enabled": false, "outputs": ["out->f.src"]}
        }
    })");

    EXPECT_EQ(stored_connections(graph->store_to_json()),
              (std::set<std::string>{"a.out->a.prev", "f.out->f.prev"}));
    EXPECT_EQ(graph->get_node_state("f"), NodeState::DISABLED);
}

TEST_F(GraphConnectionsTest, DisabledNodeWithoutSourceIsNotBypassed) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "f": {"type": "Feedback Test", "enabled": false, "outputs": ["out->b.src"]},
            "b": {"type": "Feedback Test"}
        }
    })");

    EXPECT_EQ(graph->get_node_state("f"), NodeState::DISABLED);
    EXPECT_EQ(graph->get_node_state("b"), NodeState::OK);
}

TEST_F(GraphConnectionsTest, InvalidResolvedConnectionsKeepWiring) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "x": {"type": "Buffer Test", "outputs": ["buf->f.src"]},
            "f": {"type": "Feedback Test", "enabled": false,
                  "outputs": ["out->r1.src", "out->r2.src", "out->b.src"]},
            "r1": {"type": "Required Test"},
            "r2": {"type": "Required Test"},
            "b": {"type": "Feedback Test"}
        }
    })");
    graph->request_reconnect();
    run(graph);

    EXPECT_EQ(graph->get_node_state("r1"), NodeState::ERRORED);
    EXPECT_EQ(graph->get_node_state("r2"), NodeState::ERRORED);
    EXPECT_EQ(graph->get_node_state("b"), NodeState::OK);
    EXPECT_TRUE(has_connection(graph, "f", "out->r1.src"));
    EXPECT_TRUE(has_connection(graph, "f", "out->r2.src"));
    EXPECT_TRUE(has_connection(graph, "f", "out->b.src"));
}

TEST_F(GraphConnectionsTest, InvalidDirectConnectionIsRemoved) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "x": {"type": "Buffer Test", "outputs": ["buf->b.src"]},
            "b": {"type": "Feedback Test"}
        }
    })");

    EXPECT_FALSE(has_connection(graph, "x", "buf->b.src"));
    EXPECT_EQ(graph->get_node_state("b"), NodeState::OK);
}

TEST_F(GraphConnectionsTest, DelayedReadThroughBypassOfPersistentOutputErrors) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "color": {"type": "Color", "outputs": ["out->f.src"]},
            "f": {"type": "Feedback Test", "enabled": false, "outputs": ["out->r.prev"]},
            "r": {"type": "Missing Feedback Test"}
        }
    })");

    EXPECT_EQ(graph->get_node_state("r"), NodeState::ERRORED);
    EXPECT_TRUE(has_connection(graph, "f", "out->r.prev"));
}

TEST_F(GraphConnectionsTest, UndelayedCycleErrors) {
    const GraphHandle graph = create_graph(R"({
        "version": 4,
        "nodes": {
            "x": {"type": "Feedback Test", "outputs": ["out->y.src"]},
            "y": {"type": "Feedback Test", "outputs": ["out->x.src"]}
        }
    })");

    EXPECT_EQ(graph->get_node_state("x"), NodeState::ERRORED);
    EXPECT_EQ(graph->get_node_state("y"), NodeState::ERRORED);
}
