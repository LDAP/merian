#include "graph_test.hpp"

#include "merian-graph/nodes/host_sink/host_buffer_sink.hpp"
#include "merian-graph/nodes/host_sink/host_image_sink.hpp"
#include "merian-graph/nodes/host_source/host_buffer_source.hpp"
#include "merian-graph/nodes/host_source/host_image_source.hpp"
#include "merian/io/image_io.hpp"
#include "merian/utils/defer.hpp"
#include "merian/utils/input_listener.hpp"
#include "merian/vk/window/host_window.hpp"

#include <fmt/format.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>

using namespace merian;

class HostNodes : public GraphTest {
  protected:
    static void SetUpTestSuite() {
        create_context("test-host-nodes");
    }

    static GraphDescription source_to_sink() {
        GraphDescription description;
        description.add_node("Host Image Source", "source");
        description.add_node("Host Image Sink", "sink");
        description.add_connection("source", "sink", "out", "src");
        return description;
    }

    static void upload(HostImageSource& source, const std::array<float, 8>& texels) {
        source.upload(vk::Format::eR32G32B32A32Sfloat, {2, 1, 1},
                      [&](void* data) { std::memcpy(data, texels.data(), sizeof(texels)); });
    }

    static std::vector<uint64_t>
    run_taking(const GraphHandle& graph, HostSink& sink, const uint64_t iterations) {
        std::vector<uint64_t> taken;
        const auto take = [&] {
            for (const HostSink::Capture& capture : sink.get_all()) {
                taken.push_back(capture.run_iteration);
            }
        };
        std::ignore =
            graph->run({.iterations = iterations}, {}, [&](const GraphIterationInfo&) { take(); });
        graph->wait();
        take();
        return taken;
    }

    static std::size_t waiting(HostSink& sink) {
        return sink.get_all().size();
    }
};

TEST_F(HostNodes, SourceToSink) {
    const GraphHandle graph = create_graph(source_to_sink());
    const auto source = graph->find_node_for_identifier_and_type<HostImageSource>("source");
    const auto sink = graph->find_node_for_identifier_and_type<HostImageSink>("sink");

    const std::array<float, 8> uploaded = {1, 2, 3, 4, 5, 6, 7, 8};
    upload(*source, uploaded);
    sink->capture(1);
    std::ignore = graph->run({.iterations = 2});
    graph->wait();

    const auto capture = sink->get(std::chrono::nanoseconds{0});
    ASSERT_TRUE(capture);
    EXPECT_TRUE(capture->ready());
    EXPECT_EQ(capture->copy.extent, (vk::Extent3D{2, 1, 1}));
    EXPECT_EQ(read_texels(*capture), uploaded);
    EXPECT_FALSE(sink->get(std::chrono::nanoseconds{0}));
}

TEST_F(HostNodes, RequestAt) {
    const GraphHandle graph = create_graph(color_to_sink());
    const auto sink = graph->find_node_for_identifier_and_type<HostImageSink>("sink");

    sink->capture_at({3, 5});
    std::ignore = graph->run({.iterations = 8});
    graph->wait();
    const std::vector<HostSink::Capture> captures = sink->get_all();
    ASSERT_EQ(captures.size(), 2);
    EXPECT_EQ(captures[0].run_iteration, 3);
    EXPECT_EQ(captures[1].run_iteration, 5);

    sink->capture_at({1});
    std::ignore = graph->run({.iterations = 2});
    graph->wait();
    EXPECT_EQ(waiting(*sink), 0);
}

TEST_F(HostNodes, RequestedCapturesAreNeverSkipped) {
    const GraphHandle graph = create_graph(color_to_sink());
    const auto sink = graph->find_node_for_identifier_and_type<HostImageSink>("sink");

    sink->capture(6);
    std::ignore = graph->run({.iterations = 8});
    graph->wait();
    EXPECT_EQ(waiting(*sink), 6);
}

TEST_F(HostNodes, EveryDeliversAllWhenTakenBetweenIterations) {
    const GraphHandle graph = create_graph(color_to_sink());
    const auto sink = graph->find_node_for_identifier_and_type<HostImageSink>("sink");

    sink->capture_every(1);
    const std::vector<uint64_t> taken = run_taking(graph, *sink, 50);
    ASSERT_EQ(taken.size(), 50);
    for (std::size_t i = 1; i < taken.size(); i++) {
        EXPECT_EQ(taken[i], taken[i - 1] + 1);
    }

    sink->capture_every(3);
    const std::vector<uint64_t> every_third = run_taking(graph, *sink, 9);
    ASSERT_EQ(every_third.size(), 3);
    for (const uint64_t iteration : every_third) {
        EXPECT_EQ(iteration % 3, 0);
    }
}

TEST_F(HostNodes, EverySkipsWhileCapturesWait) {
    const GraphHandle graph = create_graph(color_to_sink());
    const auto sink = graph->find_node_for_identifier_and_type<HostImageSink>("sink");

    constexpr uint32_t iterations_in_flight = 2;
    sink->capture_every(1);
    std::ignore = graph->run({.iterations = 10});
    graph->wait();
    EXPECT_EQ(waiting(*sink), iterations_in_flight + 1);

    sink->capture_every(0);
    std::ignore = graph->run({.iterations = 2});
    graph->wait();
    EXPECT_EQ(waiting(*sink), 0);
}

TEST_F(HostNodes, HeldCaptureKeepsItsTexels) {
    const GraphHandle graph = create_graph(source_to_sink());
    const auto source = graph->find_node_for_identifier_and_type<HostImageSource>("source");
    const auto sink = graph->find_node_for_identifier_and_type<HostImageSink>("sink");

    const std::array<float, 8> first = {1, 1, 1, 1, 1, 1, 1, 1};
    upload(*source, first);
    sink->capture(1);
    std::ignore = graph->run({.iterations = 1});
    const auto held = sink->get();
    ASSERT_TRUE(held);

    // later captures cycle the sink's buffers
    const std::array<float, 8> second = {2, 2, 2, 2, 2, 2, 2, 2};
    upload(*source, second);
    sink->capture_every(1);
    std::ignore = run_taking(graph, *sink, 20);
    sink->capture_every(0);
    sink->capture(1);
    std::ignore = graph->run({.iterations = 1});
    const auto latest = sink->get();

    EXPECT_EQ(read_texels(*held), first);
    ASSERT_TRUE(latest);
    EXPECT_EQ(read_texels(*latest), second);
}

TEST_F(HostNodes, CancelEndsWaits) {
    const GraphHandle graph = create_graph(color_to_sink());
    const auto sink = graph->find_node_for_identifier_and_type<HostImageSink>("sink");

    sink->capture(1);
    std::ignore = graph->run({.iterations = 1});
    bool waited_capture = false;
    bool ended = false;
    std::thread consumer([&] {
        waited_capture = sink->get().has_value();
        ended = !sink->get().has_value();
    });
    sink->cancel();
    consumer.join();
    EXPECT_TRUE(waited_capture);
    EXPECT_TRUE(ended);
}

TEST_F(HostNodes, RemovalEndsWaits) {
    for (const bool destroy_graph : {false, true}) {
        GraphHandle graph = create_graph(color_to_sink());
        const auto sink = graph->find_node_for_identifier_and_type<HostImageSink>("sink");
        std::thread consumer([&] { EXPECT_FALSE(sink->get().has_value()); });
        if (destroy_graph) {
            graph.reset();
        } else {
            graph->remove_node("sink");
        }
        consumer.join();
        EXPECT_TRUE(sink->is_cancelled());
    }
}

TEST_F(HostNodes, BufferSourceToSink) {
    GraphDescription description;
    description.add_node("Host Buffer Source", "source");
    description.add_node("Host Buffer Sink", "sink", {{"offset", 4}, {"size", 8}});
    description.add_connection("source", "sink", "out", "src");
    const GraphHandle graph = create_graph(description);
    const auto source = graph->find_node_for_identifier_and_type<HostBufferSource>("source");
    const auto sink = graph->find_node_for_identifier_and_type<HostBufferSink>("sink");

    const std::array<uint32_t, 4> uploaded = {1, 2, 3, 4};
    source->upload(sizeof(uploaded),
                   [&](void* data) { std::memcpy(data, uploaded.data(), sizeof(uploaded)); });
    sink->capture();
    std::ignore = graph->run({.iterations = 2});

    const auto capture = sink->get();
    ASSERT_TRUE(capture);
    ASSERT_EQ(capture->copy.size, 8);
    std::array<uint32_t, 2> bytes;
    capture->copy.memory->invalidate();
    std::memcpy(bytes.data(), capture->copy.memory->map(), sizeof(bytes));
    capture->copy.memory->unmap();
    EXPECT_EQ(bytes, (std::array<uint32_t, 2>{2, 3}));
}

TEST_F(HostNodes, ImageSinkWritesEveryIteration) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        fmt::format("merian-test-image-sink-{:08x}", std::random_device{}());
    defer {
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
    };
    for (const int async_tasks : {1, 3}) {
        GraphDescription description;
        description.add_node("Color", "color",
                             {{"extent", {512, 512, 1}}, {"color", {0.25, 0.5, 0.75, 1}}});
        description.add_node(
            "Image Sink", "sink",
            {{"format", "PNG"},
             {"filename",
              (directory / fmt::format("{}_{{image_index_total}}", async_tasks)).string()},
             {"enable", true},
             {"iteration", 1},
             {"iteration offset", 1}});
        description.set_graph_properties({{"max async tasks", async_tasks}});
        description.add_connection("color", "sink", "out", "src");
        const GraphHandle graph = create_graph(description);
        std::ignore = graph->run({.iterations = 16});
        graph->wait();

        // complete once wait() returned
        for (uint32_t i = 0; i < 16; i++) {
            const std::filesystem::path path = directory / fmt::format("{}_{}.png", async_tasks, i);
            ImageInfo info;
            ASSERT_NO_THROW(std::ignore = image_load_f32(path, info)) << path;
            EXPECT_EQ(info.width, 512);
            EXPECT_EQ(info.height, 512);
        }
        EXPECT_FALSE(
            std::filesystem::exists(directory / fmt::format("{}_{}.png", async_tasks, 16)));
    }
}

TEST(HostWindow, CursorInWindowCoordinates) {
    struct CursorRecorder : public InputListener {
        bool
        on_cursor(InputController& /*controller*/, const double xpos, const double ypos) override {
            cursor = {xpos, ypos};
            return true;
        }
        std::pair<double, double> cursor;
    };
    const auto window = std::make_shared<HostWindow>();
    const auto recorder = std::make_shared<CursorRecorder>();
    window->add_listener(recorder);
    window->set_extent({200, 100}, 2.f);
    window->send_cursor(100, 50);
    window->poll_events();
    EXPECT_EQ(recorder->cursor, std::make_pair(50.0, 25.0));
}
