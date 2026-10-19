import argparse
import math
import sys
from pathlib import Path

import numpy as np

import merian
import merian.graph as mg

EXAMPLES = Path(__file__).resolve().parents[1]

ACCUMULATED = "accumulate.out"
CAPTURE = "script_capture"
METRICS = ("rmse", "relmse", "mae")

_GRAPHS = {
    ".gltf": "gltf.json",
    ".glb": "gltf.json",
    ".fbx": "fbx.json",
    ".pbrt": "pbrt.json",
}
_WINDOW_NODES = ("Window", "Swapchain Blit", "ImGui")


def _graph_for(scene: Path) -> Path:
    graph = _GRAPHS.get(scene.suffix.lower())
    if graph is None:
        raise SystemExit(
            f"no graph for '{scene.suffix}' scenes, known: {', '.join(_GRAPHS)}"
        )
    return EXAMPLES / graph


def graph_and_args(scene: Path) -> tuple[Path, list[str]]:
    """Returns the graph config of a scene and its arguments; a .json is the graph."""
    if scene.suffix.lower() == ".json":
        return scene, []
    return _graph_for(scene), [str(scene.resolve())]


def parse(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> tuple[argparse.Namespace, mg.config.Config | None]:
    """
    Parses the options of a parser with a positional "scene" and of the scene's graph.
    Returns them and the composed config, None without a scene.
    """
    argv = sys.argv[1:] if argv is None else argv
    known, _ = mg.config.parse_known(parser, argv)
    if known.scene is None:
        parser.add_argument(
            "--help", action="help", help="show this help message and exit"
        )
        return parser.parse_args(argv), None
    scene = Path(known.scene)
    try:
        if scene.suffix.lower() == ".json":
            config = mg.config.Config(scene)
        else:
            config = mg.config.Config(_graph_for(scene))
            config.set(
                config.aliases()[mg.config.POSITIONAL]["pointer"], str(scene.resolve())
            )
        return mg.config.apply_args(config, argv, parser), config
    except mg.ConfigError as error:
        parser.error(str(error))


def headless(description: mg.Description, capture: str | None = None) -> mg.Description:
    """Removes the window nodes and captures "node.output" in the sink CAPTURE."""
    for identifier, node_type in description.nodes.items():
        if node_type in _WINDOW_NODES:
            description.remove_node(identifier)
    if capture:
        description.add_image_sink(capture, CAPTURE)
    return description


def displayed(description: mg.Description) -> str:
    """Returns the output ("node.output") the window shows."""
    nodes = description.nodes
    for src, dst, src_output, dst_input in description.connections:
        if nodes[dst] == "Swapchain Blit" and dst_input == "src":
            return f"{src}.{src_output}"
    raise SystemExit("the graph shows no output in a window")


def context(extensions: tuple[str, ...] = ()) -> merian.Context:
    return merian.Context(
        extensions=[*mg.EXTENSIONS, *extensions], search_paths=mg.default_search_dirs()
    )


def render(graph: mg.Graph, iterations: int) -> np.ndarray:
    """Runs `iterations` iterations and returns the last one's CAPTURE as RGB float32."""
    last = graph.run_iteration + iterations - 1
    graph.sink(CAPTURE).capture_at([last])
    graph.run(iterations=iterations)
    graph.wait()
    captures = graph.sink(CAPTURE).get_all()
    if not captures:
        raise SystemExit(
            f"nothing captured at run iteration {last}: {graph.inactive_nodes}"
        )
    return rgb(captures[-1].array)


def rgb(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[..., :3], dtype=np.float32)


def render_ms(entries: list[merian.ProfilerReport.Entry]) -> float:
    """Returns the mean GPU time of the render node, NaN if the report has none."""
    for entry in entries:
        if entry.name.startswith("render ("):
            return entry.duration_ms
        found = render_ms(entry.children)
        if not math.isnan(found):
            return found
    return float("nan")


def metrics(reference: np.ndarray, image: np.ndarray) -> dict[str, float]:
    difference = np.nan_to_num(image - reference, nan=0.0, posinf=0.0, neginf=0.0)
    epsilon = max((0.05 * float(np.mean(reference))) ** 2, 1e-12)
    squared = difference * difference
    return {
        "rmse": float(np.sqrt(np.mean(squared))),
        "relmse": float(np.mean(squared / (reference * reference + epsilon))),
        "mae": float(np.mean(np.abs(difference))),
    }
