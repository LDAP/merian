import faulthandler
import math
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path

import numpy as np

import merian
import merian.graph as mg
import merian.io


_PBRT_SCENE = """
LookAt 0 1 5  0 1 0  0 1 0
Camera "perspective" "float fov" 40
Film "rgb" "integer xresolution" 64 "integer yresolution" 64
WorldBegin
Material "diffuse" "rgb reflectance" [0.7 0.7 0.7]
Shape "trianglemesh" "integer indices" [0 1 2 0 2 3]
    "point3 P" [-1 0 -1  -1 0 1  1 0 1  1 0 -1]
AttributeBegin
    AreaLightSource "diffuse" "rgb L" [10 10 10]
    Translate 0 2 0
    Shape "sphere" "float radius" 0.25
AttributeEnd
"""


class _PathRecorder(merian.Properties):
    def __init__(self, path: str | None = None):
        self.paths: dict[str, str] = {}
        self.path = path

    def config_path(self, name, path, desc):
        self.paths[name] = path
        return self.path


class _Failing(merian.Properties):
    def __init__(self, graph: mg.Graph):
        self.graph = graph

    def config_bool(self, name, value, desc):
        self.graph.connect()


class GraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = merian.Context(extensions=mg.EXTENSIONS)

    def color_graph(self) -> mg.Graph:
        description = mg.Description()
        description.add_node("Color", "color", {"extent": [4, 2, 1]})
        description.add_image_sink("color.out", "sink")
        return mg.Graph(self.context, description)

    def test_nodes_and_edits(self):
        with self.color_graph() as graph:
            graph.connect()
            sink = graph.nodes["sink"]
            self.assertEqual((sink.identifier, sink.type), ("sink", "Host Image Sink"))
            self.assertIn("src", [connector.name for connector in sink.inputs])
            self.assertEqual(sink.connections, {"src": ("color", "out")})
            outputs = {
                output.name: output.kind for output in graph.nodes["color"].outputs
            }
            self.assertEqual(outputs["out"], mg.ConnectorKind.IMAGE)

            graph.remove_connection("color", "sink", "src")
            self.assertTrue(graph.needs_reconnect)
            graph.connect()
            self.assertEqual(sink.connections, {})
            self.assertNotEqual(sink.state, mg.NodeState.OK)

            graph.add_connection("color", "sink", "out", "src")
            graph.connect()
            self.assertEqual(sink.state, mg.NodeState.OK)
            added = graph.add_node("Color")
            self.assertIn(added, graph.nodes)
            with self.assertRaises(ValueError):
                graph.add_node("Color", "broken", {"no such property": 1})
            self.assertNotIn("broken", graph.nodes)
            node = graph.nodes[added]
            graph.remove_node(added)
            with self.assertRaises(KeyError):
                node.state

    def test_image_sink_converts_and_fits(self):
        description = mg.Description()
        description.add_node(
            "Color",
            "color",
            {
                "color": [0.25, 0.5, 0.75, 1],
                "format": "R32G32B32A32Sfloat",
                "extent": [4, 2, 1],
            },
        )
        description.add_image_sink("color.out", "sink")
        with mg.Graph(self.context, description) as graph:
            graph.nodes["sink"].set_properties(
                {"format": "R8G8B8A8Srgb", "fit": [2, 2]}
            )
            sink = graph.sink("sink")
            sink.capture()
            graph.run()
            graph.wait()
            texels = sink.get(timedelta(0)).array
            self.assertEqual((texels.shape, texels.dtype), ((1, 2, 4), np.uint8))
            # sRGB of 0.25, 0.5, 0.75
            np.testing.assert_allclose(texels[0, 0], [137, 188, 225, 255], atol=1)

    def test_buffer_sink(self):
        description = mg.Description()
        description.add_node("Color", "color", {"extent": [4, 2, 1]})
        description.add_node("Exposure", "exposure")
        description.add_connection("color", "exposure", "out", "src")
        description.add_buffer_sink("exposure.histogram", "sink")
        with mg.Graph(self.context, description) as graph:
            graph.connect()
            outputs = {
                output.name: output.kind for output in graph.nodes["exposure"].outputs
            }
            self.assertEqual(outputs["histogram"], mg.ConnectorKind.BUFFER)
            sink = graph.sink("sink")
            sink.capture()
            graph.run()
            graph.wait()
            capture = sink.get(timedelta(0))
            self.assertEqual((capture.format, capture.extent), (None, None))
            self.assertEqual(
                (capture.array.dtype, capture.array.shape),
                (np.uint8, (capture.nbytes,)),
            )
            graph.nodes["sink"].set_properties({"offset": 4, "size": 8})
            sink.capture()
            graph.run()
            graph.wait()
            self.assertEqual(sink.get(timedelta(0)).nbytes, 8)
            # A capture waits while the range holds no bytes
            graph.nodes["sink"].set_properties({"offset": 1 << 20})
            sink.capture()
            graph.run()
            graph.wait()
            self.assertEqual(sink.get_all(), [])
            graph.nodes["sink"].set_properties({"offset": 0})
            graph.run()
            graph.wait()
            self.assertEqual(sink.get(timedelta(0)).nbytes, 8)

    def test_buffer_source(self):
        description = mg.Description()
        description.add_node("Host Buffer Source", "source")
        description.add_buffer_sink("source.out", "sink")
        with mg.Graph(self.context, description) as graph:
            graph.source("source").upload(np.arange(4, dtype=np.uint32))
            sink = graph.sink("sink")
            sink.capture()
            graph.run(iterations=2)
            graph.wait()
            np.testing.assert_array_equal(
                sink.get(timedelta(0)).array.view(np.uint32), [0, 1, 2, 3]
            )

    def test_run_time(self):
        with self.color_graph() as graph:
            graph.run()
            graph.run(iterations=2, time=3600.25)
            self.assertEqual(graph.time, 3600.25)
            self.assertEqual(
                graph.to_description().graph_properties["time source"], "Auto"
            )
            start = time.monotonic()
            graph.run()
            self.assertGreaterEqual(graph.time, 3600.25)
            self.assertLessEqual(graph.time, 3600.25 + time.monotonic() - start)

    def test_empty_graph_runs(self):
        with mg.Graph(self.context) as graph:
            # A time read from the graph is set back exactly
            graph.run(time=1.001)
            self.assertEqual(graph.time, 1.001)
            graph.run(time=2)
            self.assertEqual(graph.time, 2)

    def test_infinity_is_no_limit(self):
        with self.color_graph() as graph:
            result = graph.run(iterations=3, duration=math.inf)
            self.assertEqual(
                (result.stop_reason, result.iterations),
                (mg.StopReason.ITERATIONS, 3),
            )
            with self.assertRaises(OverflowError):
                graph.run(time=math.inf)
            sink = graph.sink("sink")
            captures: list[mg.Capture | None] = []
            waiter = threading.Thread(
                target=lambda: captures.append(sink.get(math.inf))
            )
            waiter.start()
            sink.capture()
            graph.run()
            waiter.join(timeout=10)
            self.assertEqual(len(captures), 1)
            self.assertIsNotNone(captures[0])

    def test_close_in_a_with_block(self):
        with self.color_graph() as graph:
            graph.run()
            graph.close()

    def test_graph_properties(self):
        with self.color_graph() as graph:
            graph.set_properties({"graph_properties": {"iterations in flight": 3}})
            self.assertEqual(
                graph.get_properties()["graph_properties"]["iterations in flight"], 3
            )
            with self.assertRaises(ValueError):
                graph.set_properties({"graph_properties": {"iterations": 3}})

    def test_node_properties_paths(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        image = root / "image.pfm"
        merian.io.save_image(image, np.ones((2, 4, 3), np.float32))
        description = mg.Description()
        description.add_node(
            "HDR Image Source", "image", {"path": str(root / "missing.pfm")}
        )
        with mg.Graph(self.context, description) as graph:
            node = graph.nodes["image"]
            recorder = _PathRecorder()
            node.properties(recorder)
            self.assertEqual(recorder.paths["path"], str(root / "missing.pfm"))

            node.properties(_PathRecorder(str(image)))
            self.assertEqual(node.get_properties()["path"], str(image))
            self.assertTrue(graph.needs_reconnect)

    def test_ui_errors_raise_after_the_pass(self):
        with self.color_graph() as graph:
            graph.connect()
            with self.assertRaisesRegex(RuntimeError, "recording properties"):
                graph.nodes["color"].properties(_Failing(graph))
            self.assertEqual(graph.nodes["color"].state, mg.NodeState.OK)

    def test_removing_a_sink_ends_its_waits(self):
        with self.color_graph() as graph:
            graph.connect()
            sink = graph.sink("sink")
            captures: list[mg.Capture | None] = []
            waiter = threading.Thread(target=lambda: captures.append(sink.get()))
            waiter.start()
            graph.remove_node("sink")
            waiter.join(timeout=5)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(captures, [None])

    def test_recording_properties_while_a_thread_captures(self):
        with self.color_graph() as graph:
            graph.connect()
            sink = graph.sink("sink")
            done = threading.Event()

            def capture():
                while not done.is_set():
                    sink.capture()
                    sink.get_all()

            thread = threading.Thread(target=capture, daemon=True)
            thread.start()
            # A deadlock holds the GIL, the watchdog ends the process with the stacks
            faulthandler.dump_traceback_later(30, exit=True)
            try:
                for _ in range(10):
                    graph.nodes["sink"].properties(merian.Properties())
            finally:
                faulthandler.cancel_dump_traceback_later()
                done.set()
                thread.join()

    def test_capture_callbacks_in_iteration_order(self):
        description = mg.Description()
        description.add_node("Color", "color", {"extent": [4, 2, 1]})
        description.add_image_sink("color.out", "a")
        description.add_image_sink("color.out", "b")
        with mg.Graph(self.context, description) as graph:
            delivered: list[tuple[str, int]] = []

            def on_a(capture):
                delivered.append(("a", capture.run_iteration))
                graph.nodes["color"].set_properties(
                    {"color": [capture.run_iteration, 0, 0, 1]}
                )

            graph.sink("a").on_capture(on_a)
            graph.sink("b").on_capture(
                lambda capture: delivered.append(("b", capture.run_iteration))
            )
            graph.sink("a").capture_every(1)
            graph.sink("b").capture_every(2)
            graph.run(iterations=6)
            graph.wait()
            iterations = [iteration for _, iteration in delivered]
            self.assertEqual(iterations, sorted(iterations))
            self.assertEqual([name for name, _ in delivered].count("a"), 6)
            self.assertEqual([name for name, _ in delivered].count("b"), 3)

    def test_other_threads_while_running(self):
        with self.color_graph() as graph:
            running = threading.Event()
            errors: list[RuntimeError] = []

            def other():
                running.wait(timeout=5)
                try:
                    graph.connect()
                except RuntimeError as error:
                    errors.append(error)
                graph.stop()

            sink = graph.sink("sink")
            sink.on_capture(lambda _: running.set())
            sink.capture()
            thread = threading.Thread(target=other)
            thread.start()
            result = graph.run(iterations=None)
            thread.join(timeout=5)
            self.assertEqual(result.stop_reason, mg.StopReason.REQUESTED)
            self.assertEqual(len(errors), 1)

    def test_strict_load_raises_config_error(self):
        for properties in ({"unknown": 1}, {"extent": "large"}):
            description = mg.Description()
            description.add_node("Color", "color", properties)
            with self.assertRaisesRegex(mg.ConfigError, "node 'color'"):
                mg.Graph(self.context, description)
            with mg.Graph(self.context, description, strict=False) as graph:
                self.assertEqual(list(graph.nodes), ["color"])

    def test_load_rebinds_sinks(self):
        with self.color_graph() as graph:
            sink = graph.sink("sink")
            graph.load(graph.to_description())
            sink.capture()
            graph.run()
            graph.wait()
            self.assertIsNotNone(sink.get(timedelta(0)))
            self.assertIsNone(sink.get(0))

    def test_image_source_pads_rgb(self):
        description = mg.Description()
        description.add_node("Host Image Source", "source")
        description.add_image_sink("source.out", "sink")
        with mg.Graph(self.context, description) as graph:
            texels = np.random.default_rng(2).random((2, 3, 3), dtype=np.float32)
            graph.source("source").upload(texels)
            sink = graph.sink("sink")
            sink.capture()
            graph.run()
            graph.wait()
            array = sink.get(timedelta(0)).array
            np.testing.assert_array_equal(array[..., :3], texels)
            np.testing.assert_array_equal(array[..., 3], 1)

    def test_restart_run(self):
        with self.color_graph() as graph:
            graph.run(iterations=3)
            graph.restart_run()
            sink = graph.sink("sink")
            sink.capture()
            graph.run()
            graph.wait()
            capture = sink.get(timedelta(0))
            self.assertEqual(capture.run_iteration, 0)

    def test_restart_run_from_a_capture_callback(self):
        with self.color_graph() as graph:
            iterations: list[int] = []

            def on_capture(capture):
                iterations.append(capture.run_iteration)
                if len(iterations) == 3:
                    graph.restart_run()

            sink = graph.sink("sink")
            sink.on_capture(on_capture)
            sink.capture_every(1)
            graph.run(iterations=8)
            graph.wait()
            self.assertEqual(iterations[:3], [0, 1, 2])
            self.assertEqual(iterations.count(0), 2)

    def test_reports(self):
        with self.color_graph() as graph:
            graph.profiler.enabled = True
            graph.run(iterations=3)
            graph.wait()
            report = graph.profiler.report()
            self.assertIsInstance(report, merian.ProfilerReport)
            self.assertEqual(set(report.to_json()), {"cpu", "gpu"})
            memory = graph.memory_report()
            self.assertGreater(memory.total_size, 0)
            self.assertIn(memory.to_text(mg.MemoryReport.View.SUMMARY), str(memory))

    def test_description_connections(self):
        description = mg.Description()
        description.add_node("Color", "color")
        description.add_image_sink("color.out", "sink")
        self.assertEqual(description.connections, [("color", "sink", "out", "src")])

    def test_description_sink_of_a_missing_node(self):
        description = mg.Description()
        description.add_node("Color", "color")
        with self.assertRaisesRegex(ValueError, "no node 'colr'"):
            description.add_image_sink("colr.out", "sink")
        self.assertEqual(description.nodes, {"color": "Color"})
        description.add_image_sink("color.out", "sink")
        self.assertEqual(description.connections, [("color", "sink", "out", "src")])

    def path_traced(self) -> mg.Description:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        scene_file = Path(directory.name) / "scene.pbrt"
        scene_file.write_text(_PBRT_SCENE)
        description = mg.Description()
        description.add_node("PBRT Scene", "scene", {"file": str(scene_file)})
        description.add_node("GBuffer (Raytraced)", "gbuffer")
        description.add_node("Render (Path-traced)", "render")
        description.add_node("Accumulate", "accumulate")
        for src, dst, src_output, dst_input in (
            ("scene", "gbuffer", "scene", "scene"),
            ("scene", "render", "scene", "scene"),
            ("gbuffer", "render", "gbuffer", "gbuffer"),
            ("gbuffer", "accumulate", "gbuffer", "gbuffer"),
            ("render", "accumulate", "irradiance", "src"),
            ("accumulate", "accumulate", "history", "prev_history"),
            ("accumulate", "accumulate", "out", "prev_out"),
        ):
            description.add_connection(src, dst, src_output, dst_input)
        description.add_image_sink("accumulate.out", "sink")
        return description

    def render_bits(self, graph: mg.Graph) -> np.ndarray:
        sink = graph.sink("sink")
        sink.capture_at([graph.run_iteration + 7])
        graph.run(iterations=8)
        graph.wait()
        if graph.inactive_nodes:
            self.skipTest(f"inactive: {graph.inactive_nodes}")
        image = np.array(sink.get(timedelta(0)).array)
        self.assertGreater(float(image[..., :3].mean()), 0)
        return image.view(np.uint32)

    def test_fresh_graphs_render_the_same_bits(self):
        description = self.path_traced()
        with mg.Graph(self.context, description) as graph:
            first = self.render_bits(graph)
        with mg.Graph(self.context, description) as graph:
            np.testing.assert_array_equal(first, self.render_bits(graph))

    def test_restart_run_renders_the_same_bits(self):
        with mg.Graph(self.context, self.path_traced()) as graph:
            first = self.render_bits(graph)
            graph.restart_run()
            np.testing.assert_array_equal(first, self.render_bits(graph))

    def test_graph_seed_changes_the_image(self):
        with mg.Graph(self.context, self.path_traced()) as graph:
            first = self.render_bits(graph)
            graph.set_properties({"graph_properties": {"seed": 1}})
            self.assertTrue((self.render_bits(graph) != first).any())
            graph.set_properties({"graph_properties": {"seed": 0}})
            np.testing.assert_array_equal(first, self.render_bits(graph))

    def test_end_of_run_writes_the_last_iteration(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        description = mg.Description()
        description.add_node("Color", "color", {"extent": [4, 2, 1]})
        description.add_node(
            "Image Sink",
            "image",
            {
                "filename": str(Path(directory.name) / "{run_iteration}"),
                "trigger": "end of run",
                "enable": True,
            },
        )
        description.add_connection("color", "image", "out", "src")
        with mg.Graph(self.context, description) as graph:
            graph.run(iterations=5)
            graph.run(iterations=3)
            graph.run(iterations=None, duration=0.1)
        written = sorted(int(path.stem) for path in Path(directory.name).iterdir())
        self.assertEqual(written, [4, 7])

    def test_image_sink_exit_ends_the_run(self):
        description = mg.Description()
        description.add_node("Color", "color", {"extent": [4, 2, 1]})
        description.add_node("Image Sink", "image", {"advanced": {"exit at run": 2}})
        description.add_connection("color", "image", "out", "src")
        with mg.Graph(self.context, description) as graph:
            result = graph.run(iterations=10, until_event="//end")
        self.assertEqual(
            (result.stop_reason, result.iterations, result.event),
            (mg.StopReason.EVENT, 3, "Image Sink/image/end"),
        )

    def test_description_without_version_keeps_connections(self):
        description = mg.Description.from_dict(
            {
                "nodes": {
                    "color": {"type": "Color", "outputs": ["out->sink.src"]},
                    "sink": {"type": "Host Image Sink"},
                }
            }
        )
        self.assertEqual(description.connections, [("color", "sink", "out", "src")])


class HostWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = merian.Context(
            extensions=[*mg.EXTENSIONS, mg.HOST_WINDOWS_EXTENSION]
        )

    def frame_of(self, graph: mg.Graph, window: mg.HostWindow) -> mg.HostWindow.Frame:
        for _ in range(100):
            graph.run()
            frame = window.get_new_frame()
            if frame and frame.extent == window.extent:
                return frame
            time.sleep(0.01)
        self.fail("no frame arrived")

    def test_presents_into_host_memory(self):
        description = mg.Description()
        description.add_node(
            "Color", "color", {"color": [1, 0.5, 0, 1], "extent": [4, 2, 1]}
        )
        description.add_node("Window", "window")
        description.add_connection("color", "window", "out", "src")
        with mg.Graph(self.context, description) as graph:
            self.assertIsNone(graph.window("window"))
            graph.connect()
            window = graph.window("window")
            titles = {
                identifier: window.title for identifier, window in graph.windows.items()
            }
            self.assertEqual(titles, {"window": "merian: window"})
            # Nothing is presented at 0x0
            graph.run(iterations=2)
            graph.wait()
            self.assertIsNone(window.get_new_frame())

            window.set_extent(8, 4)
            frame = self.frame_of(graph, window)
            self.assertEqual((frame.extent, frame.format), ((8, 4), "R8G8B8A8Srgb"))
            # sRGB of 1, 0.5, 0
            np.testing.assert_allclose(frame.array[2, 3], [255, 188, 0, 255], atol=1)

            window.set_extent(6, 2)
            self.assertEqual(self.frame_of(graph, window).array.shape, (2, 6, 4))

    def test_close_request_removes_the_window(self):
        description = mg.Description()
        description.add_node("Window", "window")
        with mg.Graph(self.context, description) as graph:
            graph.connect()
            window = graph.window("window")
            window.request_close()
            graph.run()
            self.assertNotIn("window", graph.nodes)

    def test_close_request_sends_close(self):
        description = mg.Description()
        description.add_node(
            "Window", "window", {"on close": {"send event": True, "remove node": False}}
        )
        with mg.Graph(self.context, description) as graph:
            graph.connect()
            graph.window("window").request_close()
            result = graph.run(iterations=5, until_event="//close")
            self.assertEqual(
                (result.stop_reason, result.iterations, result.event),
                (mg.StopReason.EVENT, 1, "Window/window/close"),
            )
            self.assertIn("window", graph.nodes)


class IoTest(unittest.TestCase):
    def test_image_round_trip(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        texels = np.random.default_rng(1).random((3, 5, 3), dtype=np.float32)
        merian.io.save_image(root / "texels.pfm", texels)
        np.testing.assert_array_equal(merian.io.load_image(root / "texels.pfm"), texels)


if __name__ == "__main__":
    unittest.main()
