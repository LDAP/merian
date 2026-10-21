#include "graph_test.hpp"

#include "merian-graph/connectors/image/vk_image_in.hpp"
#include "merian-graph/connectors/image/vk_image_out_managed.hpp"
#include "merian-graph/graph/node_registry.hpp"
#include "merian-graph/nodes/compute_node/compute_node.hpp"
#include "merian-graph/nodes/host_sink/host_image_sink.hpp"
#include "merian/shader/slang_global_session.hpp"
#include "merian/utils/defer.hpp"

#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <thread>
#include <utility>
#include <vector>

using namespace merian;
using namespace std::chrono_literals;

namespace {

class GateNode : public Node {
  public:
    NodeStatusFlags process([[maybe_unused]] const NodeIO& io,
                            const NodeProcessInfo& info,
                            Submission& submission) override {
        submission.add_wait_semaphore(gate, vk::PipelineStageFlagBits::eAllCommands,
                                      info.get_total_iteration() + 1);
        return {};
    }

    static inline TimelineSemaphoreHandle gate;
};

class ElapsedNode : public Node {
  public:
    NodeStatusFlags process([[maybe_unused]] const NodeIO& io,
                            const NodeProcessInfo& info,
                            [[maybe_unused]] Submission& submission) override {
        elapsed_since_connect.push_back(info.get_elapsed_since_connect_duration());
        return {};
    }

    std::vector<std::chrono::nanoseconds> elapsed_since_connect;
};

class ThrowNode : public Node {
  public:
    std::vector<InputConnectorDescriptor> describe_inputs() override {
        return {{.name = "src", .connector = con_src, .access = ConnectorAccess::compute_read}};
    }

    NodeStatusFlags process([[maybe_unused]] const NodeIO& io,
                            [[maybe_unused]] const NodeProcessInfo& info,
                            [[maybe_unused]] Submission& submission) override {
        if (std::exchange(fail, false)) {
            throw std::runtime_error{"recording fails"};
        }
        return {};
    }

    VkImageInHandle con_src = VkImageIn::create();
    static inline bool fail = false;
};

BufferHandle create_host_buffer(const ResourceAllocatorHandle& allocator) {
    return allocator->create_buffer(sizeof(uint32_t),
                                    vk::BufferUsageFlagBits::eTransferSrc |
                                        vk::BufferUsageFlagBits::eTransferDst,
                                    MemoryMappingType::HOST_ACCESS_RANDOM, "test host buffer");
}

uint32_t read(const BufferHandle& buffer) {
    buffer->get_memory()->invalidate();
    const uint32_t value = *buffer->get_memory()->map_as<uint32_t>();
    buffer->get_memory()->unmap();
    return value;
}

void write(const BufferHandle& buffer, const uint32_t value) {
    *buffer->get_memory()->map_as<uint32_t>() = value;
    buffer->get_memory()->flush();
    buffer->get_memory()->unmap();
}

void transfer_to_host(const CommandBufferHandle& cmd, const BufferHandle& buffer) {
    cmd->barrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
        buffer->buffer_barrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead));
}

class ReadbackNode : public Node {
  public:
    void initialize([[maybe_unused]] const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override {
        this->allocator = allocator;
    }

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 [[maybe_unused]] const NodeIO& io,
                                 const NodeConnectionInfo& info,
                                 [[maybe_unused]] Submission& submission) override {
        buffers.clear();
        for (uint32_t i = 0; i < info.iterations_in_flight; i++) {
            buffers.push_back(create_host_buffer(allocator));
        }
        return {};
    }

    NodeStatusFlags process([[maybe_unused]] const NodeIO& io,
                            const NodeProcessInfo& info,
                            Submission& submission) override {
        iteration_semaphore = info.get_iteration_semaphore();
        const uint64_t iteration = info.get_total_iteration();
        const BufferHandle& buffer = buffers[info.get_in_flight_index()];
        submission.get_cmd()->fill(buffer, static_cast<uint32_t>(iteration));
        transfer_to_host(submission.get_cmd(), buffer);
        submission.sync_to_cpu(
            [this, buffer, iteration] { first.emplace_back(iteration, read(buffer)); });
        submission.sync_to_cpu([this, iteration] { second.push_back(iteration); });
        return {};
    }

    ResourceAllocatorHandle allocator;
    TimelineSemaphoreHandle iteration_semaphore;
    std::vector<BufferHandle> buffers;
    std::vector<std::pair<uint64_t, uint32_t>> first;
    std::vector<uint64_t> second;
};

// Round trips also end on_connected and process.
class RoundTripNode : public Node {
  public:
    void initialize([[maybe_unused]] const ContextHandle& context,
                    const ResourceAllocatorHandle& allocator) override {
        this->allocator = allocator;
    }

    NodeStatusFlags on_connected([[maybe_unused]] const NodeIOLayout& io_layout,
                                 [[maybe_unused]] const NodeIO& io,
                                 [[maybe_unused]] const NodeConnectionInfo& info,
                                 Submission& submission) override {
        submission.sync_to_cpu_and_back([this] { connected++; });
        return {};
    }

    NodeStatusFlags process([[maybe_unused]] const NodeIO& io,
                            const NodeProcessInfo& info,
                            Submission& submission) override {
        const uint64_t iteration = info.get_total_iteration();
        const BufferHandle shared = create_host_buffer(allocator);
        const BufferHandle result = create_host_buffer(allocator);
        submission.get_cmd()->fill(shared, static_cast<uint32_t>(iteration));
        transfer_to_host(submission.get_cmd(), shared);
        submission.sync_to_cpu_and_back([shared] { write(shared, (2 * read(shared)) + 1); });
        submission.get_cmd()->barrier(vk::PipelineStageFlagBits::eHost,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      shared->buffer_barrier(vk::AccessFlagBits::eHostWrite,
                                                             vk::AccessFlagBits::eTransferRead));
        submission.get_cmd()->copy(shared, result);
        transfer_to_host(submission.get_cmd(), result);
        submission.sync_to_cpu(
            [this, result, iteration] { results.emplace_back(iteration, read(result)); });
        submission.sync_to_cpu_and_back([] {});
        return {};
    }

    ResourceAllocatorHandle allocator;
    uint32_t connected = 0;
    std::vector<std::pair<uint64_t, uint32_t>> results;
};

class AsyncNode : public Node {
  public:
    NodeStatusFlags process([[maybe_unused]] const NodeIO& io,
                            [[maybe_unused]] const NodeProcessInfo& info,
                            Submission& submission) override {
        for (uint32_t i = 0; i < 2; i++) {
            submission.sync_to_cpu(
                [] {
                    const uint32_t now = ++running;
                    uint32_t seen = most_running.load();
                    while (now > seen && !most_running.compare_exchange_weak(seen, now)) {
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds{10});
                    running--;
                    finished++;
                },
                Submission::CallbackMode::ASYNC);
        }
        return {};
    }

    static inline std::atomic_uint32_t running = 0;
    static inline std::atomic_uint32_t most_running = 0;
    static inline std::atomic_uint32_t finished = 0;
};

class ShaderFileNode : public AbstractCompute {
  public:
    std::vector<OutputConnectorDescriptor>
    describe_outputs([[maybe_unused]] const NodeIOLayout& io_layout) override {
        return {{"out", ManagedVkImageOut::create(vk::Format::eR32G32B32A32Sfloat, {2, 1, 1}),
                 ConnectorAccess::compute_write}};
    }

    SlangCompositionHandle create_composition() override {
        const SlangCompositionHandle composition = SlangComposition::create();
        composition->add_module_from_path(file, true);
        return composition;
    }

    std::tuple<uint32_t, uint32_t, uint32_t>
    get_group_count([[maybe_unused]] const NodeIO& io) const noexcept override {
        return {1, 1, 1};
    }

    static void write_file(const float value) {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        out << fmt::format("RWTexture2D<float4> out_out;\n"
                           "[shader(\"compute\")]\n"
                           "[numthreads(2, 1, 1)]\n"
                           "void main(uint2 ipos: SV_DispatchThreadID) {{\n"
                           "    out_out[ipos] = float4({:.1f});\n"
                           "}}\n",
                           value);
    }

    static inline std::filesystem::path file;
};

} // namespace

class GraphRun : public GraphTest {
  protected:
    static void SetUpTestSuite() {
        NodeRegistry& registry = NodeRegistry::get_instance();
        registry.register_node_type<GateNode>("Gate Test", "test-graph-run");
        registry.register_node_type<ElapsedNode>("Elapsed Test", "test-graph-run");
        registry.register_node_type<ThrowNode>("Throw Test", "test-graph-run");
        registry.register_node_type<ReadbackNode>("Readback Test", "test-graph-run");
        registry.register_node_type<RoundTripNode>("Round Trip Test", "test-graph-run");
        registry.register_node_type<AsyncNode>("Async Test", "test-graph-run");
        registry.register_node_type<ShaderFileNode>("Shader File Test", "test-graph-run");
        create_context("test-graph-run");
        graph = create_graph(
            nlohmann::json::parse(R"({"version": 4, "nodes": {"color": {"type": "Color"}}})"));
    }

    static void TearDownTestSuite() {
        graph->wait();
        graph.reset();
        GateNode::gate.reset();
        GraphTest::TearDownTestSuite();
    }

    static float capture_first_channel(const GraphHandle& sink_graph) {
        const auto sink = sink_graph->find_node_for_identifier_and_type<HostImageSink>("sink");
        sink->capture(1);
        std::ignore = sink_graph->run({.iterations = 1});
        const std::optional<HostSink::Capture> capture = sink->get(10s);
        if (!capture) {
            ADD_FAILURE() << "nothing captured";
            return std::numeric_limits<float>::quiet_NaN();
        }
        return read_texels(*capture)[0];
    }

    static inline GraphHandle graph;
};

TEST_F(GraphRun, Iterations) {
    const GraphRunResult result = graph->run({.iterations = 5});
    EXPECT_EQ(result.stop_reason, GraphRunStopReason::ITERATIONS);
    EXPECT_EQ(result.iterations, 5);
}

TEST_F(GraphRun, Requested) {
    uint64_t calls = 0;
    const GraphRunResult result = graph->run(
        {.iterations = 10}, {}, [&](const GraphIterationInfo&) { calls++; },
        [&] { return calls == 2; });
    EXPECT_EQ(result.stop_reason, GraphRunStopReason::REQUESTED);
    EXPECT_EQ(result.iterations, 2);
}

TEST_F(GraphRun, EventSentBetweenIterations) {
    uint64_t calls = 0;
    const GraphRunResult result = graph->run({.iterations = 10, .until_event = "/user/done"}, {},
                                             [&](const GraphIterationInfo&) {
                                                 if (++calls == 3) {
                                                     graph->send_event("done");
                                                 }
                                             });
    EXPECT_EQ(result.stop_reason, GraphRunStopReason::EVENT);
    EXPECT_EQ(result.iterations, 3);
    EXPECT_EQ(result.event, "/user/done");
}

TEST_F(GraphRun, EventBeforeRunIsIgnored) {
    graph->send_event("early");
    const GraphRunResult result = graph->run({.iterations = 2, .until_event = "//early"});
    EXPECT_EQ(result.stop_reason, GraphRunStopReason::ITERATIONS);
}

TEST_F(GraphRun, PatternParts) {
    const auto send_done = [](const GraphIterationInfo&) { graph->send_event("done"); };
    EXPECT_EQ(graph->run({.iterations = 2, .until_event = "Color//done,//other"}, {}, send_done)
                  .stop_reason,
              GraphRunStopReason::ITERATIONS);
    EXPECT_EQ(
        graph->run({.iterations = 2, .until_event = "//other,,//done"}, {}, send_done).stop_reason,
        GraphRunStopReason::EVENT);
}

TEST_F(GraphRun, IterationCallbacks) {
    std::vector<uint64_t> before;
    std::vector<uint64_t> after;
    const GraphRunResult result = graph->run(
        {.iterations = 3},
        [&](const GraphIterationInfo& info) -> std::optional<std::chrono::nanoseconds> {
            before.push_back(info.index_in_run);
            return 10s * (info.index_in_run + 1);
        },
        [&](const GraphIterationInfo& info) {
            after.push_back(info.index_in_run);
            EXPECT_EQ(info.process_info.get_elapsed_duration(), 10s * (info.index_in_run + 1));
        });
    EXPECT_EQ(result.iterations, 3);
    EXPECT_EQ(before, (std::vector<uint64_t>{0, 1, 2}));
    EXPECT_EQ(after, (std::vector<uint64_t>{0, 1, 2}));
}

TEST_F(GraphRun, Duration) {
    const GraphRunResult result = graph->run({.iterations = std::nullopt, .duration = 0ns});
    EXPECT_EQ(result.stop_reason, GraphRunStopReason::DURATION);
    EXPECT_EQ(result.iterations, 1);
}

TEST_F(GraphRun, RunTime) {
    const GraphHandle timed = create_graph(
        nlohmann::json::parse(R"({"version": 4, "nodes": {"color": {"type": "Color"}}})"));
    std::ignore = timed->run();
    std::ignore = timed->run({.iterations = 2, .time = 3600s + 1ns});
    EXPECT_EQ(timed->get_time(), 3600s + 1ns);
    EXPECT_EQ(timed->to_description().to_json().at("graph_properties").at("time source"), "Auto");

    // the wall clock continues from the time of the run
    std::this_thread::sleep_for(200ms);
    std::ignore = timed->run();
    EXPECT_GE(timed->get_time(), 3600s + 1ns);
    EXPECT_LT(timed->get_time(), 3600s + 100ms);

    timed->set_properties({{"graph_properties", {{"time source", "Delta"}, {"delta", 10}}}});
    std::ignore = timed->run({.time = 1s});
    std::ignore = timed->run();
    EXPECT_NEAR(std::chrono::duration<double>(timed->get_time()).count(), 1.01, 1e-6);
    timed->wait();
}

TEST_F(GraphRun, RunStartsAtZeroWhateverTheTime) {
    const GraphHandle timed = create_graph(
        nlohmann::json::parse(R"({"version": 4, "nodes": {"elapsed": {"type": "Elapsed Test"}}})"));
    const auto node = timed->find_node_for_identifier_and_type<ElapsedNode>("elapsed");
    std::ignore = timed->run({.iterations = 1, .time = 5s});
    std::ignore = timed->run({.iterations = 1, .time = 6s});
    timed->restart_run();
    std::ignore = timed->run({.iterations = 1, .time = 9s});
    std::ignore = timed->run({.iterations = 1, .time = 10s});
    timed->wait();

    EXPECT_EQ(node->elapsed_since_connect, (std::vector<std::chrono::nanoseconds>{0s, 1s, 0s, 1s}));
}

TEST_F(GraphRun, DumpingPropertiesKeepsTheTime) {
    const GraphHandle timed = create_graph(
        nlohmann::json::parse(R"({"version": 4, "graph_properties": {"time source": "Time"},
            "nodes": {"color": {"type": "Color"}}})"));
    // 37 ns off the nearest float
    std::ignore = timed->run({.iterations = 1, .time = 1s + 37ns});
    std::ignore = timed->get_properties();
    std::ignore = timed->run({.iterations = 1});
    EXPECT_EQ(timed->get_time(), 1s + 37ns);

    // 1000 / (1000 / 13.f) is not 13.f
    timed->set_properties({{"graph_properties", {{"time source", "Delta"}, {"delta", 13}}}});
    std::ignore = timed->get_properties();
    std::ignore = timed->run({.iterations = 1});
    EXPECT_EQ(timed->get_time(), 1s + 37ns + 13ms);
    timed->wait();
}

TEST_F(GraphRun, AtLeastOneIterationInFlight) {
    const GraphHandle single = create_graph(nlohmann::json::parse(
        R"({"version": 4, "graph_properties": {"iterations in flight": 0, "max async tasks": 0},
            "nodes": {"color": {"type": "Color"}}})"));
    const auto in_flight = [&] {
        return single->get_properties().at("graph_properties").at("iterations in flight");
    };
    EXPECT_EQ(in_flight(), 1);
    EXPECT_EQ(single->get_properties().at("graph_properties").at("max async tasks"), 1);
    EXPECT_EQ(single->run({.iterations = 3}).iterations, 3);

    single->set_properties({{"graph_properties", {{"iterations in flight", 2}}}});
    std::ignore = single->run({.iterations = 1});
    single->set_properties({{"graph_properties", {{"iterations in flight", 0}}}});
    EXPECT_EQ(in_flight(), 1);
    EXPECT_EQ(single->run({.iterations = 3}).iterations, 3);
    single->wait();
}

TEST_F(GraphRun, SetProperties) {
    const nlohmann::json in_flight =
        graph->get_properties().at("graph_properties").at("iterations in flight");
    graph->set_properties({{"graph_properties", {{"iterations in flight", 3}}}});
    EXPECT_EQ(graph->get_properties().at("graph_properties").at("iterations in flight"), 3);
    EXPECT_THROW(graph->set_properties({{"graph_properties", {{"iterations", 3}}}}),
                 std::invalid_argument);
    EXPECT_THROW(graph->set_properties(nlohmann::json::array({1, 2})), std::invalid_argument);
    graph->set_properties({{"graph_properties", {{"iterations in flight", in_flight}}}});
}

TEST_F(GraphRun, LoadResetsSettings) {
    const GraphHandle loaded = create_graph(nlohmann::json::parse(
        R"({"version": 4, "graph_properties": {"iterations in flight": 3, "fps limiter": true},
            "profiler": {"report interval": 200}, "nodes": {"color": {"type": "Color"}}})"));
    EXPECT_EQ(loaded->get_properties().at("graph_properties").at("iterations in flight"), 3);
    loaded->from_description(GraphDescription::from_json(
        nlohmann::json::parse(R"({"version": 4, "nodes": {"color": {"type": "Color"}}})")));
    const nlohmann::json properties = loaded->get_properties();
    EXPECT_EQ(properties.at("graph_properties").at("iterations in flight"), 2);
    EXPECT_EQ(properties.at("graph_properties").at("fps limiter"), false);
    EXPECT_EQ(properties.at("profiler").at("report interval"), 50);
}

TEST_F(GraphRun, InvalidPattern) {
    EXPECT_THROW(graph->run({.iterations = 1, .until_event = "done"}), std::invalid_argument);
}

TEST_F(GraphRun, InvalidEventName) {
    EXPECT_THROW(graph->send_event("a/b"), std::invalid_argument);
    EXPECT_THROW(graph->send_event(""), std::invalid_argument);
}

TEST_F(GraphRun, SetNodeProperties) {
    graph->node_set_properties("color", {{"extent", {8, 8, 1}}});
    EXPECT_EQ(graph->node_get_properties("color").at("extent"), nlohmann::json({8, 8, 1}));
    EXPECT_TRUE(graph->get_needs_reconnect());

    EXPECT_THROW(graph->node_set_properties("color", {{"colour", 1}}), std::invalid_argument);
    EXPECT_THROW(graph->node_set_properties("color", {{"format", "nope"}}), std::invalid_argument);
    EXPECT_THROW(graph->node_set_properties("color", {{"extent", "wide"}}), std::invalid_argument);
    EXPECT_THROW(graph->node_set_properties("nope", nlohmann::json::object()),
                 std::invalid_argument);
}

TEST_F(GraphRun, CapturesShareTheIterationSemaphoreAcrossReconnects) {
    const GraphHandle color = create_graph(color_to_sink());
    const auto sink = color->find_node_for_identifier_and_type<HostImageSink>("sink");
    sink->capture(4);
    std::ignore = color->run({.iterations = 2});
    color->restart_run();
    std::ignore = color->run({.iterations = 2});
    color->wait();

    const std::vector<HostSink::Capture> captures = sink->get_all();
    ASSERT_EQ(captures.size(), 4);
    for (uint32_t i = 0; i < captures.size(); i++) {
        EXPECT_EQ(captures[i].semaphore, captures.front().semaphore);
        EXPECT_EQ(captures[i].semaphore_value, captures.front().semaphore_value + i);
    }
    EXPECT_EQ(captures.back().run_iteration, 1);
}

TEST_F(GraphRun, ConnectPicksUpEditedShaders) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() /
        fmt::format("merian-graph-shader-{:08x}", std::random_device{}());
    std::filesystem::create_directories(dir);
    defer {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    };
    ShaderFileNode::file = dir / "value.slang";

    GraphDescription description;
    description.add_node("Shader File Test", "shader");
    description.add_node("Host Image Sink", "sink");
    description.add_connection("shader", "sink", "out", "src");

    ShaderFileNode::write_file(1);
    EXPECT_EQ(capture_first_channel(create_graph(description)), 1.f);
    ShaderFileNode::write_file(2);
    const GraphHandle edited = create_graph(description);
    EXPECT_EQ(capture_first_channel(edited), 2.f) << "a new graph";
    ShaderFileNode::write_file(3);
    edited->restart_run();
    EXPECT_EQ(capture_first_channel(edited), 3.f) << "a reconnect";
    ShaderFileNode::write_file(1);
    EXPECT_EQ(capture_first_channel(create_graph(description)), 1.f) << "reverted";
}

TEST_F(GraphRun, ReduceKeepsTheShaderSession) {
    GraphDescription description;
    description.add_node(
        "Color", "color",
        {{"color", {0.25, 0.5, 0.75, 1}}, {"format", "R32G32B32A32Sfloat"}, {"extent", {2, 1, 1}}});
    description.add_node("Reduce", "reduce",
                         {{"number inputs", 1},
                          {"initial value", "vec4(0)"},
                          {"reduction", "accumulator + 2 * current_value"}});
    description.add_node("Host Image Sink", "sink");
    description.add_connection("color", "reduce", "out", "input_0");
    description.add_connection("reduce", "sink", "out", "src");

    EXPECT_EQ(capture_first_channel(create_graph(description)), 0.5f);
    const uint64_t epoch = slang_source_epoch();
    const GraphHandle reduce = create_graph(description);
    EXPECT_EQ(capture_first_channel(reduce), 0.5f) << "a new graph";
    EXPECT_EQ(slang_source_epoch(), epoch) << "a new graph";
    reduce->node_set_properties("reduce", {{"reduction", "accumulator + 3 * current_value"}});
    EXPECT_EQ(capture_first_channel(reduce), 0.75f) << "an edited expression";
    EXPECT_EQ(slang_source_epoch(), epoch) << "an edited expression";
}

TEST_F(GraphRun, ThrowWhileRecordingStillSubmits) {
    GraphDescription description = color_to_sink();
    description.add_node("Throw Test", "throw");
    description.add_connection("color", "throw", "out", "src");
    const GraphHandle color = create_graph(description);
    const auto sink = color->find_node_for_identifier_and_type<HostImageSink>("sink");
    sink->capture(1);
    ThrowNode::fail = true;
    EXPECT_THROW(color->run(), std::runtime_error);

    const auto capture = sink->get(10s);
    ASSERT_TRUE(capture);
    EXPECT_EQ(read_texels(*capture),
              (std::array<float, 8>{0.25, 0.5, 0.75, 1, 0.25, 0.5, 0.75, 1}));

    EXPECT_EQ(color->run({.iterations = 3}).iterations, 3);
    color->wait();
}

TEST_F(GraphRun, StopWhileWaitingForTheGpu) {
    GateNode::gate = TimelineSemaphore::create(context);
    const GraphHandle gated = create_graph(
        nlohmann::json::parse(R"({"version": 4, "nodes": {"gate": {"type": "Gate Test"}}})"));
    defer {
        if (GateNode::gate->get_counter_value() < 1000) {
            GateNode::gate->signal(1000);
        }
        gated->wait();
    };

    uint32_t polls = 0;
    const auto start = std::chrono::steady_clock::now();
    const GraphRunResult result =
        gated->run({.iterations = 10}, {}, {}, [&] { return ++polls == 3; });
    EXPECT_EQ(result.stop_reason, GraphRunStopReason::REQUESTED);
    // the iterations in flight were recorded, the next one waits for the first
    EXPECT_EQ(result.iterations, 2);
    EXPECT_EQ(polls, 3);
    EXPECT_LT(std::chrono::steady_clock::now() - start, 5s);

    GateNode::gate->signal(1000);
    EXPECT_EQ(gated->run({.iterations = 3}).iterations, 3);
}

TEST_F(GraphRun, CallbacksRunInOrderBeforeTheirSlotIsReused) {
    const GraphHandle readback = create_graph(
        nlohmann::json::parse(R"({"version": 4, "nodes": {"rb": {"type": "Readback Test"}}})"));
    const auto node = readback->find_node_for_identifier_and_type<ReadbackNode>("rb");
    std::ignore = readback->run({.iterations = 20});
    readback->wait();

    ASSERT_EQ(node->first.size(), 20);
    ASSERT_EQ(node->second.size(), 20);
    for (uint64_t i = 0; i < 20; i++) {
        EXPECT_EQ(node->first[i].first, node->first.front().first + i);
        EXPECT_EQ(node->first[i].second, node->first[i].first);
        EXPECT_EQ(node->second[i], node->first[i].first);
    }
}

TEST_F(GraphRun, FinishedCallbacksRunWithoutWait) {
    const GraphHandle readback = create_graph(
        nlohmann::json::parse(R"({"version": 4, "nodes": {"rb": {"type": "Readback Test"}}})"));
    const auto node = readback->find_node_for_identifier_and_type<ReadbackNode>("rb");
    std::ignore = readback->run({.iterations = 1});
    ASSERT_TRUE(node->iteration_semaphore->wait(1, 10'000'000'000));
    // the next iteration does not reuse the slot of the first
    std::ignore = readback->run({.iterations = 1});
    ASSERT_FALSE(node->first.empty());
    EXPECT_EQ(node->first.front(), (std::pair<uint64_t, uint32_t>{0, 0}));
    readback->wait();
}

TEST_F(GraphRun, RoundTripWithinAnIteration) {
    const GraphHandle round_trip = create_graph(
        nlohmann::json::parse(R"({"version": 4, "nodes": {"rt": {"type": "Round Trip Test"}}})"));
    const auto node = round_trip->find_node_for_identifier_and_type<RoundTripNode>("rt");
    // its GPU scopes span the round trips
    round_trip->profiler_set_enabled(true);
    std::ignore = round_trip->run({.iterations = 10});
    round_trip->wait();

    EXPECT_EQ(node->connected, 1);
    ASSERT_EQ(node->results.size(), 10);
    for (const auto& [iteration, value] : node->results) {
        EXPECT_EQ(value, (2 * iteration) + 1);
    }
}

TEST_F(GraphRun, AsyncCallbacksStallTheGraphInsteadOfQueueing) {
    AsyncNode::running = 0;
    AsyncNode::most_running = 0;
    AsyncNode::finished = 0;
    const GraphHandle async = create_graph(nlohmann::json::parse(R"({"version": 4,
        "graph_properties": {"max async tasks": 2},
        "nodes": {"a": {"type": "Async Test"}, "b": {"type": "Async Test"}}})"));
    const auto start = std::chrono::steady_clock::now();
    std::ignore = async->run({.iterations = 10});
    const auto run = std::chrono::steady_clock::now() - start;
    async->wait();

    EXPECT_EQ(AsyncNode::finished, 40);
    EXPECT_EQ(AsyncNode::most_running, 2);
    // 40 tasks of 10 ms, two at a time, most of them handed over during the run
    EXPECT_GT(run, 100ms);
}
