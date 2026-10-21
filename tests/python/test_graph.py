import base64
import faulthandler
import json
import math
import struct
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


def _gltf_scene() -> str:
    """A floor, an emissive ceiling quad and a clearcoated back wall."""
    quads = [
        [(-1, 0, -1), (-1, 0, 1), (1, 0, 1), (1, 0, -1)],
        [(-0.5, 2, -0.5), (0.5, 2, -0.5), (0.5, 2, 0.5), (-0.5, 2, 0.5)],
        [(-1, 0, -1), (1, 0, -1), (1, 2, -1), (-1, 2, -1)],
    ]
    blob = b"".join(struct.pack("<3f", *p) for quad in quads for p in quad)
    blob += struct.pack("<6H", 0, 1, 2, 0, 2, 3)
    blob += struct.pack("<8f", 0, 0, 0, 1, 1, 1, 1, 0)
    accessors = [
        {
            "bufferView": i,
            "componentType": 5126,
            "count": 4,
            "type": "VEC3",
            "min": np.min(quad, axis=0).tolist(),
            "max": np.max(quad, axis=0).tolist(),
        }
        for i, quad in enumerate(quads)
    ]
    accessors.append(
        {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"}
    )
    accessors.append(
        {"bufferView": 4, "componentType": 5126, "count": 4, "type": "VEC2"}
    )
    views = [{"buffer": 0, "byteOffset": 48 * i, "byteLength": 48} for i in range(3)]
    views.append({"buffer": 0, "byteOffset": 144, "byteLength": 12})
    views.append({"buffer": 0, "byteOffset": 156, "byteLength": 32})
    data = base64.b64encode(blob).decode()
    return json.dumps(
        {
            "asset": {"version": "2.0"},
            "extensionsUsed": ["KHR_materials_clearcoat"],
            "scene": 0,
            "scenes": [{"nodes": [0, 1, 2, 3]}],
            "nodes": [
                {"mesh": 0},
                {"mesh": 1},
                {"mesh": 2},
                {"camera": 0, "translation": [0, 1, 5]},
            ],
            "cameras": [
                {"type": "perspective", "perspective": {"yfov": 0.7, "znear": 0.01}}
            ],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {"POSITION": i}
                            | ({"TEXCOORD_0": 4} if i == 0 else {}),
                            "indices": 3,
                            "material": i,
                        }
                    ]
                }
                for i in range(3)
            ],
            "materials": [
                {"pbrMetallicRoughness": {"metallicFactor": 0}},
                {
                    "emissiveFactor": [1, 1, 1],
                    "doubleSided": True,
                    "pbrMetallicRoughness": {"metallicFactor": 0},
                },
                {
                    "pbrMetallicRoughness": {"metallicFactor": 0},
                    "extensions": {"KHR_materials_clearcoat": {"clearcoatFactor": 1}},
                },
            ],
            "buffers": [
                {
                    "byteLength": len(blob),
                    "uri": f"data:application/octet-stream;base64,{data}",
                }
            ],
            "bufferViews": views,
            "accessors": accessors,
        }
    )


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

    def test_captures_from_the_after_iteration_callback(self):
        description = mg.Description()
        description.add_node("Color", "color", {"extent": [4, 2, 1]})
        description.add_image_sink("color.out", "a")
        description.add_image_sink("color.out", "b")
        with mg.Graph(self.context, description) as graph:
            delivered: dict[str, list[int]] = {"a": [], "b": []}

            def take():
                for name, iterations in delivered.items():
                    captures = graph.sink(name).get_all()
                    iterations += [capture.run_iteration for capture in captures]

            def after(info: mg.IterationInfo) -> None:
                take()
                graph.nodes["color"].set_properties(
                    {"color": [info.run_index, 0, 0, 1]}
                )

            graph.sink("a").capture_every(1)
            graph.sink("b").capture_every(2)
            graph.run(iterations=6, after_iteration=after)
            graph.wait()
            take()
            self.assertEqual(delivered, {"a": [0, 1, 2, 3, 4, 5], "b": [0, 2, 4]})

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

            thread = threading.Thread(target=other)
            thread.start()
            result = graph.run(iterations=None, after_iteration=lambda _: running.set())
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

    def test_restart_run_from_an_iteration_callback(self):
        with self.color_graph() as graph:
            sink = graph.sink("sink")
            iterations: list[int] = []

            def after(info: mg.IterationInfo) -> None:
                iterations.extend(capture.run_iteration for capture in sink.get_all())
                if info.run_index == 2:
                    graph.restart_run()

            sink.capture_every(1)
            graph.run(iterations=5, after_iteration=after)
            graph.wait()
            iterations.extend(capture.run_iteration for capture in sink.get_all())
            self.assertEqual(iterations, [0, 1, 2, 0, 1])

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

    def path_traced(self, gltf: bool = False) -> mg.Description:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        scene_file = Path(directory.name) / ("scene.gltf" if gltf else "scene.pbrt")
        scene_file.write_text(_gltf_scene() if gltf else _PBRT_SCENE)
        description = mg.Description()
        scene_type = "GLTF Scene" if gltf else "PBRT Scene"
        description.add_node(scene_type, "scene", {"file": str(scene_file)})
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

    def test_find_scene_node_and_edit_its_scene(self):
        with mg.Graph(self.context, self.path_traced()) as graph:
            graph.connect()
            with self.assertRaises(KeyError):
                graph.find(mg.SceneNode, "render")
            with self.assertRaisesRegex(KeyError, "SceneNode"):
                graph.find(mg.SceneNode, "missing")
            with self.assertRaises(TypeError):
                graph.find(int, "scene")
            scene = graph.find(mg.SceneNode, "scene")
            self.assertEqual(scene.identifier, "scene")

            camera = scene.camera
            self.assertIsInstance(camera.position, tuple)
            np.testing.assert_allclose(camera.position, (0, 1, 5), atol=1e-6)
            camera.look_at((0, 1, 4), (0, 1, 0), (0, 1, 0))
            np.testing.assert_allclose(
                scene.cameras[scene.active_camera].position, (0, 1, 4)
            )
            with self.assertRaises(IndexError):
                scene.active_camera = len(scene.cameras)

            plane = next(mesh for mesh in scene.meshes if mesh.instances)
            parent = scene.add_node(animated=True)
            moving = scene.add_node(parent=parent, animated=True)
            scene.add_mesh_instance(plane.mesh_id, moving)
            fixed = scene.add_node(animated=False)
            with self.assertRaises(ValueError):
                scene.set_transform(fixed, np.eye(4, dtype=np.float32))
            for call in (
                lambda: scene.set_transform(10**6, np.eye(4, dtype=np.float32)),
                lambda: scene.add_mesh_instance(10**6, fixed),
                lambda: scene.add_mesh_instance(plane.mesh_id, 10**6),
                lambda: scene.remove_mesh_instance(plane.mesh_id, fixed),
                lambda: scene.add_node(parent=10**6),
            ):
                with self.assertRaises(KeyError):
                    call()

            positions = np.array(
                [[-1, 0.5, 0], [1, 0.5, 0], [0, 1.5, 0]], dtype=np.float32
            )
            triangles = np.array([[0, 1, 2]], dtype=np.uint32)
            with self.assertRaises(KeyError):
                scene.add_mesh(positions, triangles, 2**15)
            uvs = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
            with self.assertRaises(ValueError):
                scene.add_mesh(positions, triangles, plane.material_id, uvs=uvs[:2])
            with self.assertRaises(ValueError):
                scene.add_mesh(positions, triangles[:0], plane.material_id)
            with self.assertRaises(ValueError):
                scene.add_mesh(positions[:0], triangles[:0], plane.material_id)
            triangle = scene.add_mesh(
                positions, triangles, plane.material_id, uvs=uvs, morphed=True
            )
            scene.add_mesh_instance(triangle, scene.add_node(animated=False))
            with self.assertRaises(ValueError):
                scene.set_vertices(plane.mesh_id, positions)

            for frame in range(3):
                transform = np.eye(4, dtype=np.float32)
                transform[1, 3] = 0.1 * frame
                scene.set_transform(parent, transform)
                scene.set_vertices(triangle, positions * (1 + 0.1 * frame))
                graph.run(iterations=1)
            graph.wait()
            self.assertEqual(graph.inactive_nodes, {})
            self.assertAlmostEqual(scene.transform(moving)[1, 3], 0.2, places=6)

            scene.remove_node(parent)
            scene.remove_mesh(triangle)
            self.assertNotIn(triangle, [mesh.mesh_id for mesh in scene.meshes])
            graph.run(iterations=1)
            graph.wait()
            self.assertEqual(graph.inactive_nodes, {})

    def test_replacing_large_meshes(self):
        with mg.Graph(self.context, self.path_traced()) as graph:
            graph.connect()
            scene = graph.find(mg.SceneNode, "scene")
            material = next(mesh for mesh in scene.meshes if mesh.instances).material_id
            node = scene.add_node(animated=False)
            triangles = np.array([[0, 1, 2]], dtype=np.uint32)
            mesh = None
            for vertex_count in (2_000_000, 2_000_000, 2_400_000, 2_800_000):
                positions = np.zeros((vertex_count, 3), dtype=np.float32)
                positions[:3] = [[-1, 0.5, 0], [1, 0.5, 0], [0, 1.5, 0]]
                if mesh is not None:
                    scene.remove_mesh(mesh)
                mesh = scene.add_mesh(positions, triangles, material)
                scene.add_mesh_instance(mesh, node)
                graph.run(iterations=1)
            graph.wait()
            self.assertEqual(graph.inactive_nodes, {})

    def test_scene_textures_and_materials(self):
        with mg.Graph(self.context, self.path_traced(gltf=True)) as graph:
            graph.connect()
            scene = graph.find(mg.SceneNode, "scene")
            plane, _, coated = sorted(scene.meshes, key=lambda mesh: mesh.mesh_id)

            def mean() -> float:
                return float(self.render_bits(graph).view(np.float32)[..., :3].mean())

            dark = mean()
            checker = np.zeros((4, 8, 4), dtype=np.uint8)
            checker[::2, ::2] = 255
            texture = scene.add_texture(checker)
            material = scene.add_material(base_color_texture=texture)
            scene.set_mesh_material(plane.mesh_id, material)
            meshes = {mesh.mesh_id: mesh for mesh in scene.meshes}
            self.assertEqual(meshes[plane.mesh_id].material_id, material)
            checkered = mean()
            self.assertLess(checkered, dark)
            scene.set_texture(texture, np.full((2, 2, 4), 255, dtype=np.uint8))
            white = mean()
            self.assertGreater(white, checkered)

            glow = scene.add_texture(
                np.full((1, 1, 4), 255, dtype=np.uint8), srgb=False, mipmaps=False
            )
            scene.set_material(
                material,
                base_color_texture=texture,
                emission=(2, 2, 2),
                emission_texture=glow,
            )
            bright = mean()
            self.assertGreater(bright, white)
            scene.remove_texture(glow)
            self.assertNotEqual(mean(), bright)
            scene.set_material(material, base_color_texture=texture, roughness=0.3)
            self.assertLess(mean(), bright)
            self.assertEqual(graph.inactive_nodes, {})

            for call in (
                lambda: scene.set_texture(glow, checker),
                lambda: scene.remove_texture(glow),
                lambda: scene.add_material(emission_texture=glow),
                lambda: scene.set_material(2**15),
                lambda: scene.set_mesh_material(10**6, material),
                lambda: scene.set_mesh_material(plane.mesh_id, 2**15),
            ):
                with self.assertRaises(KeyError):
                    call()
            with self.assertRaises(ValueError):
                scene.add_texture(np.zeros((0, 4, 4), dtype=np.uint8))
            with self.assertRaises(ValueError):
                scene.set_material(coated.material_id)
            with self.assertRaises(TypeError):
                scene.add_texture(np.zeros((2, 2, 3), dtype=np.uint8))
            with self.assertRaises(TypeError):
                scene.add_texture(np.ones((2, 2, 4), dtype=np.float32))

    def test_iteration_callbacks_edit_each_iteration(self):
        with mg.Graph(self.context, self.path_traced()) as graph:
            graph.connect()
            scene = graph.find(mg.SceneNode, "scene")
            sink = graph.sink("sink")
            sink.capture_every(1)
            times = []
            seen = []
            described = []

            def counters(info: mg.IterationInfo) -> tuple[int, int, int]:
                return (info.run_index, info.iteration, info.total_iteration)

            def before(info: mg.IterationInfo) -> float:
                described.append(counters(info))
                scene.camera.look_at((0, 1, 5 + info.run_index), (0, 1, 0), (0, 1, 0))
                return 10.0 * (info.run_index + 1)

            def after(info: mg.IterationInfo) -> None:
                self.assertEqual(counters(info), described[-1])
                times.extend(capture.time for capture in sink.get_all())
                seen.append((info.run_index, info.time, scene.camera.position[2]))

            result = graph.run(
                iterations=4, before_iteration=before, after_iteration=after
            )
            graph.wait()
            self.assertEqual(result.iterations, 4)
            self.assertEqual([index for index, _, _ in seen], [0, 1, 2, 3])
            self.assertEqual([at for _, at, _ in seen], [10.0, 20.0, 30.0, 40.0])
            np.testing.assert_allclose([z for _, _, z in seen], [5, 6, 7, 8], atol=1e-5)
            times.extend(capture.time for capture in sink.get_all())
            self.assertEqual(times, [10.0, 20.0, 30.0, 40.0])

            # the clock continues from the time held last
            graph.run(iterations=1)
            self.assertGreaterEqual(graph.time, 40.0)

            def failing(info: mg.IterationInfo) -> None:
                if info.run_index == 1:
                    raise ValueError("stop here")

            # the iteration being prepared still runs, after_iteration is not called for it
            afters = []
            start = graph.run_iteration
            with self.assertRaisesRegex(ValueError, "stop here"):
                graph.run(
                    iterations=5,
                    before_iteration=failing,
                    after_iteration=lambda info: afters.append(info.run_index),
                )
            self.assertEqual(graph.run_iteration, start + 2)
            self.assertEqual(afters, [0])
            with self.assertRaises(RuntimeError):
                graph.run(iterations=1, before_iteration=lambda _: graph.wait())
            with self.assertRaises(TypeError):
                graph.run(iterations=1, before_iteration=lambda _: "later")
            graph.wait()

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
