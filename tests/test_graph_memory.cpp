#include "graph_test.hpp"

#include "merian-graph/connectors/image/vk_image_in_sampled.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"

using namespace merian;
using json = nlohmann::json;

namespace {

class MemoryTestNode : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{"src", con_src, ConnectorAccess::compute_read, 0, true},
                {"prev", con_prev, ConnectorAccess::compute_read, 1, true}};
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {
            {"out", ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 256, 256),
             ConnectorAccess::compute_write},
            {.name = "off",
             .connector = ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 256, 256),
             .access = ConnectorAccess::compute_write,
             .disabled = true},
            {.name = "tmp",
             .connector = ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 512, 512),
             .access = ConnectorAccess::compute_read_write,
             .scratch = true},
            {.name = "state",
             .connector = ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 64, 64),
             .access = ConnectorAccess::compute_read_write,
             .scratch = true,
             .persistent = true},
        };
    }

    NodeStatusFlags process(const NodeIO& /*io*/,
                            const NodeProcessInfo& /*info*/,
                            Submission& /*submission*/) override {
        process_count++;
        return {};
    }

    uint32_t process_count = 0;

  private:
    const VkSampledImageInHandle con_src = VkSampledImageIn::create();
    const VkSampledImageInHandle con_prev = VkSampledImageIn::create();
};

class TwoInputsNode : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{"a", con_a, ConnectorAccess::compute_read},
                {"b", con_b, ConnectorAccess::compute_read},
                {"prev", con_prev, ConnectorAccess::compute_read, 1, true}};
    }

    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {{"out", ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 64, 64),
                 ConnectorAccess::compute_write}};
    }

  private:
    const VkSampledImageInHandle con_a = VkSampledImageIn::create();
    const VkSampledImageInHandle con_b = VkSampledImageIn::create();
    const VkSampledImageInHandle con_prev = VkSampledImageIn::create();
};

class DisabledAccessNode : public Node {
  public:
    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {{.name = "off",
                 .connector = con_off,
                 .access = ConnectorAccess::compute_write,
                 .disabled = true}};
    }

    NodeStatusFlags process(const NodeIO& io,
                            const NodeProcessInfo& /*info*/,
                            Submission& /*submission*/) override {
        process_count++;
        std::ignore = io[con_off];
        return {};
    }

    uint32_t process_count = 0;

  private:
    const ManagedVkImageOutHandle con_off =
        ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 16, 16);
};

// Writes its output in on_connected.
template <bool persistent> class ConnectWriterNode : public Node {
  public:
    std::vector<OutputConnectorDescriptor>
    describe_outputs(const NodeIOLayout& /*io_layout*/) override {
        return {{.name = "out",
                 .connector = con_out,
                 .access = ConnectorAccess::transfer_dst,
                 .persistent = persistent}};
    }

    NodeStatusFlags on_connected(const NodeIOLayout& /*io_layout*/,
                                 const NodeIO& io,
                                 const NodeConnectionInfo& /*info*/,
                                 Submission& submission) override {
        const ImageHandle image = io[con_out].get_image();
        submission.get_cmd()->barrier(image->barrier2(
            vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eMemoryWrite,
            vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands,
            vk::PipelineStageFlagBits2::eTransfer, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            all_levels_and_layers(), true));
        submission.get_cmd()->clear(image);
        return {};
    }

  private:
    const ManagedVkImageOutHandle con_out =
        ManagedVkImageOut::create(vk::Format::eR8G8B8A8Unorm, 16, 16);
};

json tonemap_chain(const bool alias) {
    return json::parse(fmt::format(R"({{
        "version": 4,
        "graph_properties": {{"alias transient resources": {}}},
        "nodes": {{
            "color": {{"type": "Color", "outputs": ["out->t0.src"]}},
            "t0": {{"type": "Tonemap", "outputs": ["out->t1.src"]}},
            "t1": {{"type": "Tonemap", "outputs": ["out->t2.src"]}},
            "t2": {{"type": "Tonemap", "outputs": ["out->t3.src"]}},
            "t3": {{"type": "Tonemap"}}
        }}
    }})",
                                   alias));
}

class GraphMemoryTest : public GraphTest {
  protected:
    static void SetUpTestSuite() {
        NodeRegistry::get_instance().register_node_type<MemoryTestNode>("Memory Test",
                                                                        "test-graph-memory");
        NodeRegistry::get_instance().register_node_type<DisabledAccessNode>("Disabled Access",
                                                                            "test-graph-memory");
        NodeRegistry::get_instance().register_node_type<TwoInputsNode>("Two Inputs",
                                                                       "test-graph-memory");
        NodeRegistry::get_instance().register_node_type<ConnectWriterNode<false>>(
            "Transient Connect Writer", "test-graph-memory");
        NodeRegistry::get_instance().register_node_type<ConnectWriterNode<true>>(
            "Persistent Connect Writer", "test-graph-memory");
        create_context("test-graph-memory");
    }

    using GraphTest::create_graph;

    static GraphHandle create_graph(const json& description) {
        const GraphHandle graph = create_graph();
        graph->load_from_json(description);
        run(graph);
        return graph;
    }

    static const GraphMemoryReport::Placed* find_placed(const GraphMemoryReport& report,
                                                        const std::string& output) {
        const auto it =
            std::ranges::find(report.placed, output, &GraphMemoryReport::Placed::output);
        return it == report.placed.end() ? nullptr : &*it;
    }

    static std::vector<uint32_t> positions(const GraphMemoryReport::Placed& placed) {
        std::vector<uint32_t> in_use;
        for (uint32_t position = 0; position < placed.lifetime.get_positions(); position++) {
            if (placed.lifetime.contains(position)) {
                in_use.push_back(position);
            }
        }
        return in_use;
    }
};

} // namespace

TEST_F(GraphMemoryTest, ChainPingPongs) {
    const GraphMemoryReport report = create_graph(tonemap_chain(true))->get_memory_report();

    EXPECT_EQ(report.period, 1);
    std::vector<const GraphMemoryReport::Placed*> outputs;
    for (uint32_t i = 0; i < 4; i++) {
        const auto* output = outputs.emplace_back(find_placed(report, fmt::format("t{}.out", i)));
        ASSERT_TRUE(output);
        EXPECT_EQ(positions(*output),
                  (i < 3 ? std::vector<uint32_t>{i + 1, i + 2} : std::vector<uint32_t>{i + 1}));
        EXPECT_EQ(output->placement.block_index, outputs[0]->placement.block_index);
    }
    EXPECT_EQ(outputs[0]->placement.offset, outputs[2]->placement.offset);
    EXPECT_EQ(outputs[1]->placement.offset, outputs[3]->placement.offset);
    EXPECT_NE(outputs[0]->placement.offset, outputs[1]->placement.offset);
    // the color source is persistent
    const auto* color = find_placed(report, "color.out");
    ASSERT_TRUE(color);
    EXPECT_EQ(positions(*color).size(), report.layer_count);
    EXPECT_FALSE(color->placement.aliased);
    EXPECT_TRUE(report.unplaced.empty());
    // at most two neighbors are in use with the source
    ASSERT_EQ(report.in_use.size(), report.layer_count);
    const vk::DeviceSize most_in_use = *std::ranges::max_element(report.in_use);
    EXPECT_EQ(most_in_use, color->placement.size + 2 * outputs[0]->placement.size);
    EXPECT_GE(report.blocks.at(0).size, most_in_use);
}

TEST_F(GraphMemoryTest, ChainWithoutAliasing) {
    const GraphMemoryReport report = create_graph(tonemap_chain(false))->get_memory_report();

    EXPECT_FALSE(report.aliasing);
    ASSERT_EQ(report.placed.size(), 5);
    for (const auto& output : report.placed) {
        EXPECT_FALSE(output.placement.aliased) << output.output;
    }
}

TEST_F(GraphMemoryTest, ScratchAndDisabledOutputs) {
    const GraphMemoryReport report = create_graph(json::parse(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Memory Test", "outputs": ["out->b.src"]},
            "b": {"type": "Memory Test"}
        }
    })"))
                                         ->get_memory_report();

    for (const auto& output : report.placed) {
        EXPECT_FALSE(output.output.ends_with(".off")) << output.output;
    }
    EXPECT_TRUE(report.unplaced.empty());

    const auto* a_out = find_placed(report, "a.out");
    const auto* a_tmp = find_placed(report, "a.tmp");
    const auto* b_tmp = find_placed(report, "b.tmp");
    ASSERT_TRUE(a_out && a_tmp && b_tmp);
    EXPECT_FALSE(a_out->scratch);
    EXPECT_TRUE(a_tmp->scratch);
    EXPECT_EQ(positions(*a_out), (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(positions(*a_tmp), std::vector<uint32_t>{0});
    EXPECT_EQ(positions(*b_tmp), std::vector<uint32_t>{1});
    EXPECT_EQ(a_tmp->placement.offset, b_tmp->placement.offset);

    const auto* a_state = find_placed(report, "a.state");
    ASSERT_TRUE(a_state);
    EXPECT_TRUE(a_state->scratch);
    EXPECT_EQ(positions(*a_state).size(), report.layer_count);
    EXPECT_FALSE(a_state->placement.aliased);
}

// Each copy of t.out fits into the copy of f.out that is not in use.
TEST_F(GraphMemoryTest, DelayedFeedbackSharesWithDownstream) {
    const GraphMemoryReport report = create_graph(json::parse(R"({
        "version": 4,
        "nodes": {
            "f": {"type": "Memory Test", "outputs": ["out->f.prev", "out->t.src"]},
            "t": {"type": "Memory Test", "outputs": ["out->u.src"]},
            "u": {"type": "Memory Test"}
        }
    })"))
                                         ->get_memory_report();

    ASSERT_EQ(report.layer_count, 3);
    ASSERT_EQ(report.period, 2);
    EXPECT_FALSE(find_placed(report, "f.out"));
    const auto* f_0 = find_placed(report, "f.out (0)");
    const auto* f_1 = find_placed(report, "f.out (1)");
    const auto* t_0 = find_placed(report, "t.out (0)");
    const auto* t_1 = find_placed(report, "t.out (1)");
    ASSERT_TRUE(f_0 && f_1 && t_0 && t_1);
    EXPECT_EQ(positions(*f_0), (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(positions(*f_1), (std::vector<uint32_t>{0, 3, 4, 5}));
    EXPECT_EQ(positions(*t_0), (std::vector<uint32_t>{1, 2}));
    EXPECT_EQ(positions(*t_1), (std::vector<uint32_t>{4, 5}));
    EXPECT_EQ(t_0->placement.offset, f_1->placement.offset);
    EXPECT_EQ(t_1->placement.offset, f_0->placement.offset);
}

TEST_F(GraphMemoryTest, ScratchCannotBeConnected) {
    const GraphHandle graph = create_graph(json::parse(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Memory Test", "outputs": ["tmp->b.src"]},
            "b": {"type": "Memory Test"}
        }
    })"));
    EXPECT_FALSE(graph->store_to_json()["nodes"]["a"].contains("outputs"));
}

TEST_F(GraphMemoryTest, DisabledOutputAccessStopsNode) {
    const GraphHandle graph = create_graph(json::parse(R"({
        "version": 4,
        "nodes": {"a": {"type": "Disabled Access"}, "b": {"type": "Memory Test"}}
    })"));
    const auto a = graph->find_node_for_identifier_and_type<DisabledAccessNode>("a");
    const auto b = graph->find_node_for_identifier_and_type<MemoryTestNode>("b");
    ASSERT_TRUE(a && b);
    EXPECT_EQ(a->process_count, 1);
    EXPECT_EQ(b->process_count, 3);
}

// In on_connected a node sees only copy 0 of an output, and transient contents do not survive.
TEST_F(GraphMemoryTest, OnConnectedWritesOnlyPersistentOutputs) {
    const GraphHandle graph = create_graph(json::parse(R"({
        "version": 4,
        "nodes": {
            "transient": {"type": "Transient Connect Writer", "outputs": ["out->a.prev"]},
            "persistent": {"type": "Persistent Connect Writer"},
            "a": {"type": "Memory Test"}
        }
    })"));
    EXPECT_EQ(graph->get_node_state("transient"), NodeState::ERRORED);
    EXPECT_EQ(graph->get_node_state("persistent"), NodeState::OK);
    EXPECT_TRUE(draw(graph).shows_text("not persistent", {"transient"}));
}

TEST_F(GraphMemoryTest, ForceDisabledConsumerIsNotALifetime) {
    const GraphHandle graph = create_graph(json::parse(R"({
        "version": 4,
        "nodes": {
            "a": {"type": "Memory Test", "outputs": ["out->d.a", "out->d.prev"]},
            "b": {"type": "Memory Test", "outputs": ["out->c.src"]},
            "c": {"type": "Memory Test", "outputs": ["out->d.b"]},
            "d": {"type": "Two Inputs"}
        }
    })"));
    const GraphMemoryReport delayed = graph->get_memory_report();
    ASSERT_EQ(delayed.layer_count, 3);
    EXPECT_EQ(delayed.period, 2);
    EXPECT_TRUE(find_placed(delayed, "a.out (1)"));

    // d requires an output that b disabled
    graph->add_connection("b", "d", "off", "b");
    graph->run();
    graph->wait();
    const GraphMemoryReport report = graph->get_memory_report();
    // the delayed read does not happen
    EXPECT_EQ(report.period, 1);
    const auto* a_out = find_placed(report, "a.out");
    ASSERT_TRUE(a_out);
    EXPECT_EQ(positions(*a_out), std::vector<uint32_t>{0});
    std::ignore = fmt::format("{}", report);
}

TEST_F(GraphMemoryTest, ReconnectReusesHeldMemory) {
    const GraphHandle graph = create_graph(tonemap_chain(true));
    const MemoryAllocationHandle held = graph->get_memory_report().blocks.at(0).memory.lock();
    ASSERT_TRUE(held);

    graph->request_reconnect();
    graph->run();
    graph->wait();
    EXPECT_EQ(graph->get_memory_report().blocks.at(0).memory.lock(), held);
}

TEST_F(GraphMemoryTest, ShrinkingOutputsShrinksMemory) {
    const GraphHandle graph = create_graph(tonemap_chain(true));
    const auto set_extent = [&](const uint32_t extent) {
        JSONLoadProperties props(json::parse(
            fmt::format(R"({{"nodes": {{"color": {{"properties": {{"extent": [{}, {}, 1]}}}}}}}})",
                        extent, extent)));
        graph->properties(props);
        graph->run();
        graph->wait();
        return graph->get_memory_report().blocks.at(0).size;
    };

    const vk::DeviceSize small = set_extent(256);
    const vk::DeviceSize large = set_extent(1024);
    EXPECT_GE(large, 8 * small);
    // also while something holds the large block
    const MemoryAllocationHandle held = graph->get_memory_report().blocks.at(0).memory.lock();
    EXPECT_EQ(set_extent(256), small);
    EXPECT_NE(graph->get_memory_report().blocks.at(0).memory.lock(), held);

    // reports and the memory view do not keep it alive past the iterations in flight that used it
    set_extent(1024);
    const GraphMemoryReport large_report = graph->get_memory_report();
    UIProperties props;
    graph->properties(props);
    set_extent(256);
    graph->run();
    graph->wait();
    EXPECT_TRUE(large_report.blocks.at(0).memory.expired());
}

TEST_F(GraphMemoryTest, PropertiesShowMemory) {
    const GraphHandle graph = create_graph(tonemap_chain(true));
    const vk::DeviceSize block_size = graph->get_memory_report().blocks.at(0).size;
    UIProperties props;
    graph->properties(props);
    graph->properties(props);
    graph->run();
    graph->wait();
    EXPECT_EQ(graph->get_memory_report().blocks.at(0).size, block_size);

    const GraphMemoryReport report = graph->get_memory_report();
    const std::string text = fmt::format("{}", report);
    EXPECT_NE(text.find("Block 0"), std::string::npos);
    EXPECT_NE(text.find("t3.out"), std::string::npos);
    EXPECT_TRUE(props.shows_text(report.format_summary(), {"memory"}));
    EXPECT_TRUE(props.shows_text(report.format_blocks(), {"memory", "blocks"}));
    // the overview shows what the node shows
    EXPECT_TRUE(props.shows_text(report.format_node("t3"), {"memory", "nodes", "t3"}));
    EXPECT_TRUE(props.shows_text(report.format_node("t3"), {"t3", "memory"}));
}
