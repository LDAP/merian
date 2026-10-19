#!/usr/bin/env python3
"""
A graph editor: the nodes and their connections, the properties of the selected node,
times, memory, and what the graph shows: outputs, and the windows of its Window nodes,
which take the input of their view.

  graph_editor.py [scene or graph.json] [graph options]

Right-click adds a node. Drag from an output to an input to connect them. Click an
output to preview it above the properties; right-click it to show it as the backdrop of
the graph or in a panel next to the graph. The header of each view, and View > Backdrop
for the backdrop, select what it shows. Delete removes the selected nodes and
connections. Hollow inputs are optional; grey connectors and dashed connections were
connected but are not described by their node now. Positions are stored in the saved
config.
"""

import argparse
import functools
import json
import math
import queue
import signal
import sys
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    raise SystemExit("the graph editor needs PySide6: pip install PySide6") from None

from lib import scene

# isort: split
import merian.graph as mg
from gui.output_view import (
    CaptureImage,
    OutputPanel,
    OutputView,
    Source,
    SourceKind,
    fit_properties,
    resize_timer,
    set_source_actions,
)
from gui.qt_properties import QtProperties, Recorder
from gui.report_views import (
    WIDEST_SIZE,
    MemoryView,
    ProfilerView,
    format_ms,
    format_size,
    node_times,
)
from gui.theme import (
    NODE_STATE_COLORS,
    RIGHT_ALIGNED,
    THEME,
    theme_palette,
    theme_stylesheet,
)
from gui.window_input import WindowInput

_BACKDROP = "backdrop"
_PREVIEW = "preview"
_SINK_TYPES = {
    mg.ConnectorKind.IMAGE: "Host Image Sink",
    mg.ConnectorKind.BUFFER: "Host Buffer Sink",
}
_SINK_CONTENTS = {node_type: content for content, node_type in _SINK_TYPES.items()}
_VIEW_FORMAT = "R8G8B8A8Srgb"
_VIEW_SINK = "gui view"

_CLOCK_INTERVAL = timedelta(milliseconds=33)
_CONNECTED = "/graph/connect"
# Below any refresh period, which Qt can misreport
_FRAME_INTERVAL = timedelta(milliseconds=4)
_MEMORY_INTERVAL = timedelta(seconds=1)
_PROPERTIES_INTERVAL = timedelta(milliseconds=100)
_REPORT_INTERVAL = timedelta(milliseconds=100)
_RUN_SLICE = timedelta(milliseconds=15)

_BACKDROP_SHADE = QtGui.QColor(THEME["canvas"])
_BACKDROP_SHADE.setAlpha(120)
_INPUT_COLOR = THEME["blue"]
_OUTPUT_COLORS = {
    mg.ConnectorKind.IMAGE: THEME["accent"],
    mg.ConnectorKind.BUFFER: THEME["peach"],
    mg.ConnectorKind.OTHER: THEME["text_dim"],
}
_SHOWN_COLOR = THEME["green"]
_UNDESCRIBED_COLOR = THEME["overlay"]

_CHIP_GAP = 4
_CHIP_HEIGHT = 18
_CONNECTOR_RADIUS = 5
_CORNER_RADIUS = 8
_GRID_SPACING = 24
_HEADER_HEIGHT = 28
_LAYER_SPACING = 80
_MIN_FIT_SCALE = 0.8
_NODE_SPACING = 30
_ROW_HEIGHT = 20
_SHADOW = ((5, 22), (2, 34))
_SHADOW_OFFSET = 3
# The center of the first connector row
_TITLE_HEIGHT = 44


# Nodes and Sources
###################


class _Input(NamedTuple):
    name: str
    delay: int
    optional: bool


class _Output(NamedTuple):
    name: str
    kind: mg.ConnectorKind


@dataclass
class _NodeInfo:
    node_type: str
    state: mg.NodeState
    inactive_reason: str | None
    inputs: list[_Input]
    outputs: list[_Output]
    # input -> (node, output)
    connections: dict[str, tuple[str, str]]

    def delay(self, input_name: str) -> int:
        return next((i.delay for i in self.inputs if i.name == input_name), 0)

    def output_kind(self, output: str) -> mg.ConnectorKind:
        return next(
            (o.kind for o in self.outputs if o.name == output), mg.ConnectorKind.OTHER
        )


def _can_show(view: str, source: Source) -> bool:
    return view != _BACKDROP or source.content == mg.ConnectorKind.IMAGE


class _GraphState(NamedTuple):
    """Top-level nodes; window titles by Window node identifier, subgraphs included."""

    nodes: dict[str, _NodeInfo]
    windows: dict[str, str]


def _view_sources(state: _GraphState) -> list[Source]:
    sources = [
        Source(SourceKind.WINDOW, identifier, title=title)
        for identifier, title in sorted(state.windows.items(), key=lambda item: item[1])
    ]
    for identifier, node in sorted(state.nodes.items()):
        if node.node_type in _SINK_CONTENTS:
            sources.append(
                Source(
                    SourceKind.SINK, identifier, content=_SINK_CONTENTS[node.node_type]
                )
            )
        sources += [
            Source(SourceKind.OUTPUT, identifier, output.name, output.kind)
            for output in node.outputs
            if output.kind in _SINK_TYPES
        ]
    return sources


def _read_state(graph: mg.Graph, hidden: set[str]) -> _GraphState:
    inactive = graph.inactive_nodes
    nodes = {}
    for identifier, node in graph.nodes.items():
        if identifier in hidden:
            continue
        nodes[identifier] = _NodeInfo(
            node.type,
            node.state,
            inactive.get(identifier),
            [_Input(i.name, i.delay, i.optional) for i in node.inputs],
            [_Output(o.name, o.kind) for o in node.outputs],
            node.connections,
        )
    windows = {identifier: window.title for identifier, window in graph.windows.items()}
    return _GraphState(nodes, windows)


def _undescribed_outputs(nodes: dict[str, _NodeInfo]) -> dict[str, list[str]]:
    undescribed = defaultdict(set)
    for node in nodes.values():
        for identifier, output in node.connections.values():
            if identifier in nodes and output not in {
                o.name for o in nodes[identifier].outputs
            }:
                undescribed[identifier].add(output)
    return {identifier: sorted(outputs) for identifier, outputs in undescribed.items()}


def _layer_nodes(nodes: dict[str, _NodeInfo]) -> dict[str, int]:
    layers: dict[str, int] = {}

    def layer(identifier: str, visiting: set[str]) -> int:
        if identifier in layers:
            return layers[identifier]
        if identifier in visiting:
            return 0
        visiting.add(identifier)
        node = nodes[identifier]
        sources = {
            source
            for input_name, (source, _) in node.connections.items()
            if node.delay(input_name) == 0 and source in nodes
        }
        layers[identifier] = max(
            (layer(source, visiting) + 1 for source in sources), default=0
        )
        return layers[identifier]

    for identifier in nodes:
        layer(identifier, set())
    return layers


def _stored_positions(description: mg.Description) -> dict[str, QtCore.QPointF]:
    positions = {}
    for identifier, node in description.to_dict().get("nodes", {}).items():
        if position := node.get("metadata", {}).get("position"):
            positions[identifier] = QtCore.QPointF(*position)
    return positions


# Graph Worker
##############


@dataclass
class _ViewSink:
    identifier: str
    content: mg.ConnectorKind
    owned: bool
    sink: mg.HostSink | None = None
    generation: int = 0
    requested: bool = False
    unshown: int = 0
    interval: float = 0.0
    requested_time: float = -math.inf


class _GraphWorker(QtCore.QObject):
    """Makes every graph call on its own thread; commands run between iterations."""

    nodes_changed = QtCore.Signal(object)
    node_added = QtCore.Signal(str, object)
    # sequence, identifier, items
    properties_recorded = QtCore.Signal(int, str, object)
    # iteration, time
    clocked = QtCore.Signal(object, float)
    reported = QtCore.Signal(object)
    # In ms
    profiler_interval_changed = QtCore.Signal(int)
    # view, generation, capture
    captured = QtCore.Signal(str, int, object)
    # view, generation, HostWindow or None
    window_shown = QtCore.Signal(str, int, object)
    playing_changed = QtCore.Signal(bool)
    message = QtCore.Signal(str)

    def __init__(self, graph: mg.Graph):
        super().__init__()
        self.graph = graph
        self.commands: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self.runner = threading.Thread(target=self.loop, name="graph", daemon=True)
        self.playing = True
        self.steps = 0
        self.paused_time: float | None = None
        self.quitting = False
        self.view_sinks: dict[str, _ViewSink] = {}
        # generation, Window node and its HostWindow by view
        self.view_windows: dict[str, tuple[int, str, mg.HostWindow | None]] = {}
        self.sinks_added = 0
        self.properties_request: tuple[int, str, Recorder] | None = None
        self.times_shown = False
        self.memory_shown = False
        self.report_kind = mg.ReportKind.INTERVAL
        self.profiler_interval = 0
        self.sampled_totals: tuple[float, float] | None = None
        self.next_clock = 0.0
        self.next_report = 0.0
        self.next_memory_report = 0.0

    def post(self, command: Callable[..., None], *args) -> None:
        self.commands.put(functools.partial(command, *args))

    def start(self) -> None:
        self.post(self.connect_graph)
        self.runner.start()

    def quit(self) -> None:
        self.post(setattr, self, "quitting", True)
        self.graph.stop()
        self.runner.join()

    @property
    def running(self) -> bool:
        return self.playing or self.steps > 0

    def loop(self) -> None:
        while True:
            self.run_commands(block=not self.running)
            if self.quitting:
                return
            if self.properties_request:
                request, self.properties_request = self.properties_request, None
                self.attempt(lambda: self.record_properties(*request))
            if self.running:
                self.attempt(self.run)
            self.attempt(self.take_captures)
            self.attempt(self.report)

    def attempt(self, command: Callable[[], None]) -> None:
        try:
            command()
        except Exception as error:
            traceback.print_exc()
            self.message.emit(str(error))

    def run_commands(self, block: bool) -> None:
        try:
            command = (
                self.commands.get(timeout=0.1) if block else self.commands.get_nowait()
            )
            while True:
                self.attempt(command)
                command = self.commands.get_nowait()
        except queue.Empty:
            pass

    def run(self) -> None:
        now = time.monotonic()
        for view_sink in self.view_sinks.values():
            # One shown, one waiting, one on its way at most; a step shows its iteration
            due = (
                not view_sink.requested
                and view_sink.unshown <= 1
                and now - view_sink.requested_time >= view_sink.interval
            )
            if self.steps or due:
                view_sink.sink.capture()
                view_sink.requested = True
                view_sink.requested_time = now
        if self.paused_time is not None:
            # A step and the first iteration after a pause run at the time paused at
            limits = {"iterations": 1, "time": self.paused_time}
            if self.playing:
                self.paused_time = None
        else:
            limits = {"iterations": None, "duration": _RUN_SLICE}
        self.steps = max(self.steps - 1, 0)
        try:
            result = self.graph.run(**limits, until_event=_CONNECTED)
            if result.stop_reason == mg.StopReason.EVENT:
                self.publish_state()
        except Exception:
            self.set_playing(False)
            raise

    def take_captures(self) -> None:
        """Publishes the latest captures, to be confirmed with capture_shown."""
        for view, view_sink in self.view_sinks.items():
            if captures := view_sink.sink.get_all():
                view_sink.requested = False
                view_sink.unshown += 1
                self.captured.emit(view, view_sink.generation, captures[-1])

    def report(self) -> None:
        now = time.monotonic()
        if now >= self.next_clock:
            self.next_clock = now + _CLOCK_INTERVAL.total_seconds()
            self.clocked.emit(self.graph.run_iteration, self.graph.time)
        report: dict[str, object] = {}
        profiler = self.graph.profiler
        profiled = profiler.enabled and self.times_shown
        if profiled:
            # The graph replaces the interval report once per report interval
            interval_report = profiler.report(mg.ReportKind.INTERVAL)
            totals = (interval_report.cpu_total_ms(), interval_report.gpu_total_ms())
            if totals != self.sampled_totals:
                self.sampled_totals = totals
                report["profile"] = interval_report
        if now >= self.next_report:
            self.next_report = now + max(
                _REPORT_INTERVAL.total_seconds(), self.profiler_interval / 1000
            )
            if profiled:
                report["profile shown"] = profiler.report(self.report_kind)
            if now >= self.next_memory_report:
                self.next_memory_report = now + _MEMORY_INTERVAL.total_seconds()
                memory = self.graph.memory_report()
                report["memory"] = memory.total_size
                if self.memory_shown:
                    report["memory report"] = memory.to_json()
        if report:
            self.reported.emit(report)

    def connect_graph(self) -> None:
        self.graph.connect()
        self.profiler_interval = self.graph.get_properties()["profiler"][
            "report interval"
        ]
        self.profiler_interval_changed.emit(self.profiler_interval)
        self.publish_state()

    def publish_state(self) -> None:
        nodes = self.graph.nodes
        self.view_sinks = {
            view: view_sink
            for view, view_sink in self.view_sinks.items()
            if view_sink.identifier in nodes
        }
        self.nodes_changed.emit(_read_state(self.graph, self.owned_sinks()))
        windows = self.graph.windows
        for view, (generation, identifier, window) in list(self.view_windows.items()):
            found = windows.get(identifier)
            if found is not None and found is not window:
                self.show_window(view, generation, identifier, found)

    def capture_shown(self, view: str, generation: int) -> None:
        view_sink = self.view_sinks.get(view)
        if view_sink and view_sink.generation == generation:
            view_sink.unshown -= 1

    def set_playing(self, playing: bool) -> None:
        self.playing = playing
        self.playing_changed.emit(playing)
        if not playing:
            self.graph.wait()
            self.paused_time = self.graph.time

    def step(self) -> None:
        self.steps += 1

    def stop(self) -> None:
        self.set_playing(False)
        self.graph.restart_run()

    def reload_shaders(self) -> None:
        self.graph.reload_shaders()

    def set_reports(self, times: bool, memory: bool) -> None:
        self.times_shown, self.memory_shown = times, memory
        self.next_report = self.next_memory_report = 0.0

    def set_profiling(self, enabled: bool) -> None:
        self.graph.profiler.enabled = enabled

    def set_report_kind(self, kind: mg.ReportKind) -> None:
        self.report_kind = kind
        self.next_report = 0.0

    def set_profiler_interval(self, milliseconds: int) -> None:
        try:
            self.graph.set_properties({"profiler": {"report interval": milliseconds}})
        except Exception:
            self.profiler_interval_changed.emit(self.profiler_interval)
            raise
        self.profiler_interval = milliseconds

    def restart_profiler_report(self) -> None:
        self.graph.profiler.restart_report()

    def edit(self, change: Callable[[mg.Graph], None]) -> None:
        try:
            change(self.graph)
        finally:
            if self.graph.needs_reconnect:
                self.connect_graph()

    def add_node(self, node_type: str, position: QtCore.QPointF) -> None:
        self.node_added.emit(self.graph.add_node(node_type), position)
        self.connect_graph()

    def add_connection(
        self, src: str, src_output: str, dst: str, dst_input: str
    ) -> None:
        self.edit(lambda graph: graph.add_connection(src, dst, src_output, dst_input))

    def delete(
        self, identifiers: list[str], connections: list[tuple[str, str, str]]
    ) -> None:
        def remove(graph: mg.Graph) -> None:
            for src, dst, dst_input in connections:
                graph.remove_connection(src, dst, dst_input)
            for identifier in identifiers:
                graph.remove_node(identifier)

        self.edit(remove)

    def show(
        self,
        view: str,
        generation: int,
        source: Source,
        properties: dict,
        interval: timedelta,
    ) -> None:
        if source.kind == SourceKind.WINDOW:
            self.hide(view)
            self.show_window(
                view,
                generation,
                source.identifier,
                self.graph.window(source.identifier),
            )
            return
        if source.kind == SourceKind.SINK:
            self.hide(view)
            view_sink = _ViewSink(
                source.identifier,
                source.content,
                False,
                self.graph.sink(source.identifier),
            )
        else:
            view_sink = self.capture_output(view, source, properties)
        view_sink.generation, view_sink.requested, view_sink.unshown = (
            generation,
            False,
            0,
        )
        view_sink.interval = interval.total_seconds()
        self.view_sinks[view] = view_sink
        if reason := self.graph.inactive_nodes.get(view_sink.identifier):
            self.message.emit(f"cannot show {source.label()}: {reason}")

    def show_window(
        self, view: str, generation: int, identifier: str, window: mg.HostWindow | None
    ) -> None:
        self.view_windows[view] = (generation, identifier, window)
        self.window_shown.emit(view, generation, window)
        if window is None:
            self.message.emit(f"{identifier} presents to its own window")

    def capture_output(self, view: str, source: Source, properties: dict) -> _ViewSink:
        self.view_windows.pop(view, None)
        previous = self.view_sinks.pop(view, None)
        if previous and not previous.owned:
            previous = None
        if previous and previous.content == source.content:
            view_sink = previous
        else:
            nodes = self.graph.nodes
            self.sinks_added += 1
            while f"{_VIEW_SINK} {self.sinks_added}" in nodes:
                self.sinks_added += 1
            view_sink = _ViewSink(
                f"{_VIEW_SINK} {self.sinks_added}", source.content, True
            )
        # Hidden from the state the edit publishes
        self.view_sinks[view] = view_sink

        def connect(graph: mg.Graph) -> None:
            if view_sink is not previous:
                if previous:
                    graph.remove_node(previous.identifier)
                graph.add_node(
                    _SINK_TYPES[source.content],
                    view_sink.identifier,
                    {"format": _VIEW_FORMAT}
                    if source.content == mg.ConnectorKind.IMAGE
                    else {},
                )
            try:
                graph.nodes[view_sink.identifier].set_properties(properties)
                graph.add_connection(
                    source.identifier, view_sink.identifier, source.output, "src"
                )
            except Exception:
                graph.remove_node(view_sink.identifier)
                raise

        # The reconnect completes the captures of the previous output, which are dropped
        try:
            self.edit(connect)
        except Exception:
            self.view_sinks.pop(view, None)
            raise
        if view_sink is previous:
            view_sink.sink.get_all()
        else:
            view_sink.sink = self.graph.sink(view_sink.identifier)
        return view_sink

    def owned_sinks(self) -> set[str]:
        return {
            view_sink.identifier
            for view_sink in self.view_sinks.values()
            if view_sink.owned
        }

    def hide(self, view: str) -> None:
        self.view_windows.pop(view, None)
        if (view_sink := self.view_sinks.pop(view, None)) and view_sink.owned:
            self.edit(lambda graph: graph.remove_node(view_sink.identifier))

    def set_capture_interval(self, view: str, interval: timedelta) -> None:
        if view_sink := self.view_sinks.get(view):
            view_sink.interval = interval.total_seconds()

    def set_view_properties(self, view: str, properties: dict) -> None:
        if (view_sink := self.view_sinks.get(view)) and view_sink.owned:
            self.graph.nodes[view_sink.identifier].set_properties(properties)

    def load(self, description: mg.Description) -> None:
        self.view_sinks = {}
        self.view_windows = {}
        self.edit(lambda graph: graph.load(description, strict=False))

    def save(self, path: Path, positions: dict[str, QtCore.QPointF]) -> None:
        description = self.graph.to_description()
        owned = self.owned_sinks()
        for identifier in description.nodes:
            if identifier in owned:
                description.remove_node(identifier)
        config = description.to_dict()
        for identifier, position in positions.items():
            if identifier in config.get("nodes", {}):
                metadata = config["nodes"][identifier].setdefault("metadata", {})
                metadata["position"] = [round(position.x()), round(position.y())]
        path.write_text(json.dumps(config, indent=4) + "\n")
        self.message.emit(f"saved {path}")

    def request_properties(
        self, sequence: int, identifier: str, recorder: Recorder
    ) -> None:
        if self.properties_request:
            _, pending_identifier, pending = self.properties_request
            if pending_identifier == identifier:
                recorder.changes = pending.changes | recorder.changes
            else:
                self.record_properties(*self.properties_request)
        self.properties_request = (sequence, identifier, recorder)

    def record_properties(
        self, sequence: int, identifier: str, recorder: Recorder
    ) -> None:
        nodes = self.graph.nodes
        if identifier not in nodes:
            return
        try:
            nodes[identifier].properties(recorder)
        finally:
            self.properties_recorded.emit(sequence, identifier, recorder.items)
            if self.graph.needs_reconnect:
                self.connect_graph()


# Graph View
############


@functools.cache
def _bold_font() -> QtGui.QFont:
    font = QtGui.QFont(QtWidgets.QApplication.font())
    font.setBold(True)
    return font


@functools.cache
def _font_metrics(bold: bool) -> QtGui.QFontMetricsF:
    return QtGui.QFontMetricsF(_bold_font() if bold else QtWidgets.QApplication.font())


def _text_width(text: str, bold: bool = False) -> float:
    return _font_metrics(bold).horizontalAdvance(text)


def _mixed(color: str, other: str, weight: float) -> QtGui.QColor:
    a, b = QtGui.QColor(color).getRgbF()[:3], QtGui.QColor(other).getRgbF()[:3]
    return QtGui.QColor.fromRgbF(*(x + (y - x) * weight for x, y in zip(a, b)))


class _Connector(QtWidgets.QGraphicsEllipseItem):
    def __init__(
        self,
        node: "_NodeItem",
        name: str,
        is_output: bool,
        row: int,
        optional: bool,
        described: bool,
        kind: mg.ConnectorKind,
    ):
        super().__init__(
            -_CONNECTOR_RADIUS,
            -_CONNECTOR_RADIUS,
            2 * _CONNECTOR_RADIUS,
            2 * _CONNECTOR_RADIUS,
            node,
        )
        self.node, self.name, self.is_output = node, name, is_output
        self.optional, self.described, self.kind = optional, described, kind
        self.setPos(
            node.rect().width() if is_output else 0, _TITLE_HEIGHT + _ROW_HEIGHT * row
        )
        self.set_shown(False)
        label = QtWidgets.QGraphicsSimpleTextItem(name, self)
        label.setBrush(QtGui.QColor(THEME["text"] if described else _UNDESCRIBED_COLOR))
        bounds = label.boundingRect()
        x = (
            -bounds.width() - 2 * _CONNECTOR_RADIUS
            if is_output
            else 2 * _CONNECTOR_RADIUS
        )
        label.setPos(x, -bounds.height() / 2)

    def set_shown(self, shown: bool) -> None:
        if not self.described:
            color = _UNDESCRIBED_COLOR
        elif shown:
            color = _SHOWN_COLOR
        else:
            color = _OUTPUT_COLORS[self.kind] if self.is_output else _INPUT_COLOR
        self.setPen(QtGui.QPen(QtGui.QColor(color), 2))
        self.setBrush(QtGui.QColor(THEME["surface"] if self.optional else color))


class _NodeItem(QtWidgets.QGraphicsRectItem):
    def __init__(self, identifier: str, node: _NodeInfo, extra_outputs: list[str]):
        declared = {i.name for i in node.inputs}
        other = mg.ConnectorKind.OTHER
        inputs = [(i.name, i.optional, True, other) for i in node.inputs]
        inputs += [
            (name, False, False, other)
            for name in sorted(node.connections.keys() - declared)
        ]
        outputs = [(o.name, False, True, o.kind) for o in node.outputs]
        outputs += [(name, False, False, other) for name in extra_outputs]
        self.identifier_width = _text_width(identifier, bold=True)
        connector_width = (
            max((_text_width(connector[0]) for connector in inputs), default=0)
            + max((_text_width(connector[0]) for connector in outputs), default=0)
            + 8 * _CONNECTOR_RADIUS
        )
        title_width = self.identifier_width + _text_width(f"  {node.node_type}")
        width = max(title_width + 5 * _CONNECTOR_RADIUS, connector_width)
        rows = max(len(inputs), len(outputs), 1)
        height = _TITLE_HEIGHT + _ROW_HEIGHT * (rows - 1) + _ROW_HEIGHT // 2 + 6
        super().__init__(0, 0, width, height)
        self.identifier, self.node_type = identifier, node.node_type
        # GPU ms and share of the total
        self.time: float | None = None
        self.share = 0.0
        self.state = node.state
        state_color = THEME[NODE_STATE_COLORS[node.state]]
        self.state_color = QtGui.QColor(state_color)
        self.header_color = (
            QtGui.QColor(THEME["surface_hi"])
            if node.state == mg.NodeState.OK
            else _mixed(THEME["surface_hi"], state_color, 0.35)
        )
        self.edges: list[_Edge] = []
        flags = self.GraphicsItemFlag
        self.setFlags(
            flags.ItemIsMovable
            | flags.ItemIsSelectable
            | flags.ItemSendsGeometryChanges
        )
        if node.inactive_reason:
            self.setToolTip(node.inactive_reason)
        self.inputs = {
            name: _Connector(self, name, False, row, optional, described, kind)
            for row, (name, optional, described, kind) in enumerate(inputs)
        }
        self.outputs = {
            name: _Connector(self, name, True, row, optional, described, kind)
            for row, (name, optional, described, kind) in enumerate(outputs)
        }

    def boundingRect(self) -> QtCore.QRectF:
        grow = _SHADOW[0][0]
        return self.rect().adjusted(
            -grow, -_CHIP_HEIGHT - _CHIP_GAP - 2, grow, grow + _SHADOW_OFFSET
        )

    def set_time(self, milliseconds: float | None, share: float) -> None:
        if (milliseconds, share) != (self.time, self.share):
            self.time, self.share = milliseconds, share
            rect = self.rect()
            self.update(
                rect.left(),
                rect.top() - _CHIP_HEIGHT - _CHIP_GAP,
                rect.width(),
                _CHIP_HEIGHT,
            )

    def paint(self, painter: QtGui.QPainter, _option, _widget=None) -> None:
        rect = self.rect()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for grow, alpha in _SHADOW:
            painter.setBrush(QtGui.QColor(0, 0, 0, alpha))
            painter.drawRoundedRect(
                rect.adjusted(
                    -grow, _SHADOW_OFFSET - grow, grow, _SHADOW_OFFSET + grow
                ),
                _CORNER_RADIUS + grow,
                _CORNER_RADIUS + grow,
            )
        body = QtGui.QPainterPath()
        body.addRoundedRect(rect, _CORNER_RADIUS, _CORNER_RADIUS)
        painter.fillPath(body, QtGui.QColor(THEME["surface"]))
        band = QtGui.QPainterPath()
        band.addRect(
            QtCore.QRectF(rect.left(), rect.top(), rect.width(), _HEADER_HEIGHT)
        )
        painter.fillPath(body.intersected(band), self.header_color)

        title = QtCore.QRectF(
            2 * _CONNECTOR_RADIUS,
            0,
            rect.width() - 4 * _CONNECTOR_RADIUS,
            _HEADER_HEIGHT,
        )
        align = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        painter.setFont(_bold_font())
        painter.setPen(QtGui.QColor(THEME["text"]))
        painter.drawText(title, align, self.identifier)
        painter.setFont(QtWidgets.QApplication.font())
        painter.setPen(QtGui.QColor(THEME["text_dim"]))
        painter.drawText(
            title.adjusted(self.identifier_width, 0, 0, 0), align, f"  {self.node_type}"
        )

        if self.isSelected():
            border = QtGui.QPen(QtGui.QColor(THEME["accent"]), 2)
        elif self.state == mg.NodeState.OK:
            border = QtGui.QPen(QtGui.QColor(THEME["surface_hi"]), 1)
        else:
            border = QtGui.QPen(self.state_color, 1.5)
        painter.setPen(border)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(body)

        if self.time is not None:
            text = format_ms(self.time)
            width = _text_width(text) + 14
            chip = QtCore.QRectF(
                rect.right() - width,
                rect.top() - _CHIP_HEIGHT - _CHIP_GAP,
                width,
                _CHIP_HEIGHT,
            )
            heat = min(self.share * 2, 1.0)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(_mixed(THEME["surface_hi"], THEME["red"], heat))
            painter.drawRoundedRect(chip, _CHIP_HEIGHT / 2, _CHIP_HEIGHT / 2)
            painter.setPen(QtGui.QColor(THEME["bg" if heat > 0.5 else "text"]))
            painter.setFont(QtWidgets.QApplication.font())
            painter.drawText(chip, QtCore.Qt.AlignmentFlag.AlignCenter, text)

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            for edge in self.edges:
                edge.update_path()
        return super().itemChange(change, value)


class _Edge(QtWidgets.QGraphicsPathItem):
    def __init__(self, source: _Connector, target: _Connector):
        super().__init__()
        self.source, self.target = source, target
        described = source.described and target.described
        color = QtGui.QColor(
            _OUTPUT_COLORS[source.kind] if described else _UNDESCRIBED_COLOR
        )
        color.setAlpha(200)
        pen = QtGui.QPen(color, 2)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        if not described:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        self.setPen(pen)
        self.setFlags(self.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)
        source.node.edges.append(self)
        target.node.edges.append(self)
        self.update_path()

    def paint(self, painter: QtGui.QPainter, _option, _widget=None) -> None:
        pen = QtGui.QPen(self.pen())
        if self.isSelected():
            pen.setColor(QtGui.QColor(THEME["accent_hi"]))
            pen.setWidthF(3)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())

    def update_path(self) -> None:
        start, end = self.source.scenePos(), self.target.scenePos()
        bend = QtCore.QPointF(max(abs(end.x() - start.x()) / 2, 40), 0)
        path = QtGui.QPainterPath(start)
        path.cubicTo(start + bend, end - bend, end)
        self.setPath(path)


class _NodePicker(QtWidgets.QMenu):
    picked = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.filter = QtWidgets.QLineEdit(placeholderText="add node")
        self.list = QtWidgets.QListWidget()
        self.list.addItems(sorted(mg.node_types()))
        self.list.setMinimumHeight(320)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.addWidget(self.filter)
        layout.addWidget(self.list)
        action = QtWidgets.QWidgetAction(self)
        action.setDefaultWidget(content)
        self.addAction(action)
        self.filter.textChanged.connect(self.narrow)
        self.filter.returnPressed.connect(lambda: self.pick(self.list.currentItem()))
        self.list.itemClicked.connect(self.pick)
        self.narrow("")

    def narrow(self, text: str) -> None:
        visible = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setHidden(text.lower() not in item.text().lower())
            if not item.isHidden():
                visible.append(item)
        self.list.setCurrentItem(visible[0] if visible else None)

    def pick(self, item: QtWidgets.QListWidgetItem | None) -> None:
        if item is not None:
            self.picked.emit(item.text())
        self.close()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self.filter.clear()
        self.filter.setFocus()


class _GraphView(QtWidgets.QGraphicsView):
    # src, src_output, dst, dst_input
    connection_requested = QtCore.Signal(str, str, str, str)
    # node, output
    output_clicked = QtCore.Signal(str, str)
    # node, output, global position
    output_menu_requested = QtCore.Signal(str, str, object)
    # identifiers, [(src, dst, dst_input)]
    delete_requested = QtCore.Signal(list, list)
    # identifier or None
    node_selected = QtCore.Signal(object)
    # node type, scene position
    node_requested = QtCore.Signal(str, object)
    # Properties of the backdrop's Host Sink
    sink_changed = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.items_by_identifier: dict[str, _NodeItem] = {}
        self.dragged: _Connector | None = None
        self.drag_line: QtWidgets.QGraphicsLineItem | None = None
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QtGui.QColor(THEME["canvas"]))
        self.setDragMode(self.DragMode.RubberBandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.scene().selectionChanged.connect(self.emit_selection)
        self.node_times: dict[str, float] = {}
        self.picker_position = QtCore.QPointF()
        self.picker = _NodePicker(self)
        self.picker.picked.connect(
            lambda node_type: self.node_requested.emit(node_type, self.picker_position)
        )
        self.backdrop = CaptureImage()
        self.resize_timer = resize_timer(
            self, lambda: self.sink_changed.emit(self.sink_properties())
        )
        self.input = WindowInput(
            self, lambda: self.backdrop.area(self.viewport().rect()), self.viewport()
        )

    def sink_properties(self) -> dict:
        return fit_properties(self.viewport())

    def show_capture(self, capture: mg.Capture | mg.HostWindow.Frame | None) -> None:
        self.backdrop.set_capture(capture, self.devicePixelRatioF())
        # The backdrop does not scroll with the nodes
        modes = self.ViewportUpdateMode
        self.setViewportUpdateMode(
            modes.MinimalViewportUpdate if capture is None else modes.FullViewportUpdate
        )
        self.viewport().update()

    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        super().drawBackground(painter, rect)
        if self.backdrop.image is not None:
            painter.save()
            painter.resetTransform()
            self.backdrop.draw(painter, self.viewport().rect())
            painter.fillRect(self.viewport().rect(), _BACKDROP_SHADE)
            painter.restore()
            return
        spacing = _GRID_SPACING
        while spacing * self.transform().m11() < 12:
            spacing *= 2
        left = math.floor(rect.left() / spacing) * spacing
        top = math.floor(rect.top() / spacing) * spacing
        columns = int((rect.right() - left) / spacing) + 1
        rows = int((rect.bottom() - top) / spacing) + 1
        pen = QtGui.QPen(_mixed(THEME["surface"], THEME["surface_hi"], 0.5), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPoints(
            [
                QtCore.QPointF(left + column * spacing, top + row * spacing)
                for column in range(columns)
                for row in range(rows)
            ]
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.resize_timer.start()

    def show_graph(
        self,
        nodes: dict[str, _NodeInfo],
        new_positions: dict[str, QtCore.QPointF],
        shown: set[tuple[str, str]],
        selected: str | None,
    ) -> None:
        positions = self.positions() | new_positions
        graphics = self.scene()
        graphics.blockSignals(True)
        graphics.clear()
        self.dragged = self.drag_line = None
        extra_outputs = _undescribed_outputs(nodes)
        self.items_by_identifier = {
            identifier: _NodeItem(identifier, node, extra_outputs.get(identifier, []))
            for identifier, node in nodes.items()
        }
        self.place(nodes, positions)
        for item in self.items_by_identifier.values():
            graphics.addItem(item)
        self.set_node_times(self.node_times)
        for identifier, node in nodes.items():
            for input_name, (source_identifier, output) in node.connections.items():
                if source_identifier in self.items_by_identifier:
                    source_connector = self.items_by_identifier[
                        source_identifier
                    ].outputs[output]
                    target_connector = self.items_by_identifier[identifier].inputs[
                        input_name
                    ]
                    graphics.addItem(_Edge(source_connector, target_connector))
        if selected in self.items_by_identifier:
            self.items_by_identifier[selected].setSelected(True)
        graphics.blockSignals(False)
        self.set_shown_outputs(shown)

    def set_shown_outputs(self, shown: set[tuple[str, str]]) -> None:
        for identifier, item in self.items_by_identifier.items():
            for output, connector in item.outputs.items():
                connector.set_shown((identifier, output) in shown)

    def place(
        self, nodes: dict[str, _NodeInfo], positions: dict[str, QtCore.QPointF]
    ) -> None:
        layers = _layer_nodes(nodes)
        widths = defaultdict(float)
        for identifier, item in self.items_by_identifier.items():
            widths[layers[identifier]] = max(
                widths[layers[identifier]], item.rect().width()
            )
        column_x, x = {}, 0.0
        for layer in sorted(widths):
            column_x[layer] = x
            x += widths[layer] + _LAYER_SPACING
        column_y = defaultdict(float)
        for identifier in sorted(nodes, key=lambda i: (layers[i], i)):
            item = self.items_by_identifier[identifier]
            if identifier in positions:
                item.setPos(positions[identifier])
                continue
            layer = layers[identifier]
            item.setPos(column_x[layer], column_y[layer])
            column_y[layer] += item.rect().height() + _NODE_SPACING

    def fit(self) -> None:
        bounds = self.scene().itemsBoundingRect()
        self.fitInView(bounds, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        scale = min(max(self.transform().m11(), _MIN_FIT_SCALE), 1.0)
        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(bounds.center())

    def clear_graph(self) -> None:
        self.scene().clear()
        self.items_by_identifier = {}

    def set_node_times(self, times: dict[str, float]) -> None:
        self.node_times = times
        total = sum(times.values()) or 1.0
        for identifier, item in self.items_by_identifier.items():
            time = times.get(identifier)
            item.set_time(time, (time or 0.0) / total)

    def positions(self) -> dict[str, QtCore.QPointF]:
        return {
            identifier: item.pos()
            for identifier, item in self.items_by_identifier.items()
        }

    def emit_selection(self) -> None:
        nodes = [
            item for item in self.scene().selectedItems() if isinstance(item, _NodeItem)
        ]
        self.node_selected.emit(nodes[0].identifier if nodes else None)

    def select_node(self, identifier: str) -> None:
        """Selects the node, as clicking it does."""
        if item := self.items_by_identifier.get(identifier):
            self.scene().clearSelection()
            item.setSelected(True)
            self.ensureVisible(item)

    def connector_at(self, position: QtCore.QPointF) -> _Connector | None:
        item = self.itemAt(position.toPoint())
        if item is not None and isinstance(item.parentItem(), _Connector):
            item = item.parentItem()
        return item if isinstance(item, _Connector) else None

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        connector = self.connector_at(QtCore.QPointF(event.pos()))
        if connector and connector.is_output:
            self.output_menu_requested.emit(
                connector.node.identifier, connector.name, event.globalPos()
            )
            return
        self.picker_position = self.mapToScene(event.pos())
        self.picker.exec(event.globalPos())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        connector = self.connector_at(event.position())
        if (
            event.button() == QtCore.Qt.MouseButton.LeftButton
            and connector
            and connector.is_output
        ):
            self.dragged = connector
            line = QtCore.QLineF(connector.scenePos(), connector.scenePos())
            pen = QtGui.QPen(QtGui.QColor(THEME["text_dim"]), 2)
            self.drag_line = self.scene().addLine(line, pen)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.drag_line:
            end = self.mapToScene(event.position().toPoint())
            self.drag_line.setLine(QtCore.QLineF(self.dragged.scenePos(), end))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self.drag_line:
            super().mouseReleaseEvent(event)
            return
        self.scene().removeItem(self.drag_line)
        source, self.dragged, self.drag_line = self.dragged, None, None
        target = self.connector_at(event.position())
        if target is source:
            self.output_clicked.emit(source.node.identifier, source.name)
        elif target and not target.is_output:
            self.connection_requested.emit(
                source.node.identifier, source.name, target.node.identifier, target.name
            )

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() != QtCore.Qt.Key.Key_Delete:
            super().keyPressEvent(event)
            return
        identifiers, connections = [], []
        for item in self.scene().selectedItems():
            if isinstance(item, _NodeItem):
                identifiers.append(item.identifier)
            elif isinstance(item, _Edge):
                connections.append(
                    (
                        item.source.node.identifier,
                        item.target.node.identifier,
                        item.target.name,
                    )
                )
        self.delete_requested.emit(identifiers, connections)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        factor = 1.0015 ** event.angleDelta().y()
        self.scale(factor, factor)


# Main Window
#############


def _property_pointer(key: tuple) -> str | None:
    if any(occurrence for _, _, occurrence in key):
        return None
    names = [name for _, name, _ in key]
    if len(names) == 2 and names[1] in ("enabled", "linearization order"):
        path = names
    elif len(names) > 3 and names[1:3] == ["sections", "properties"]:
        path = [names[0], "properties", *names[3:]]
    else:
        return None
    return "/nodes/" + "/".join(
        name.replace("~", "~0").replace("/", "~1") for name in path
    )


def _add_action(
    menu: QtWidgets.QMenu,
    text: str,
    triggered: Callable[..., None] | None = None,
    shortcut: QtGui.QKeySequence.StandardKey | str | None = None,
    tip: str = "",
    checkable: bool = False,
) -> QtGui.QAction:
    action = menu.addAction(text)
    action.setCheckable(checkable)
    if triggered:
        (action.toggled if checkable else action.triggered).connect(triggered)
    if shortcut is not None:
        action.setShortcut(QtGui.QKeySequence(shortcut))
    if tip:
        action.setToolTip(tip)
    return action


def _polygon(*points: tuple[float, float]) -> QtGui.QPolygonF:
    return QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])


@functools.cache
def _painted_icon(name: str) -> QtGui.QIcon:
    pixmap = QtGui.QPixmap(64, 64)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    color = QtGui.QColor(THEME["text"])
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(color)
    if name == "play":
        painter.drawPolygon(_polygon((20, 12), (52, 32), (20, 52)))
    elif name == "pause":
        painter.drawRect(QtCore.QRectF(17, 14, 11, 36))
        painter.drawRect(QtCore.QRectF(36, 14, 11, 36))
    elif name == "step":
        painter.drawPolygon(_polygon((14, 14), (40, 32), (14, 50)))
        painter.drawRect(QtCore.QRectF(42, 14, 8, 36))
    elif name == "stop":
        painter.drawRect(QtCore.QRectF(16, 16, 32, 32))
    elif name == "reconnect":
        painter.setPen(
            QtGui.QPen(
                color, 6, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap
            )
        )
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawArc(QtCore.QRectF(12, 12, 40, 40), 90 * 16, 270 * 16)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(_polygon((42, 34), (62, 34), (52, 20)))
    else:
        font = painter.font()
        font.setPixelSize(36)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(pixmap.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "{ }")
    painter.end()
    return QtGui.QIcon(pixmap)


class _Shown(NamedTuple):
    source: Source
    generation: int


class _Window(QtWidgets.QMainWindow):
    def __init__(
        self,
        worker: _GraphWorker,
        path: Path | None,
        new_positions: dict[str, QtCore.QPointF],
    ):
        super().__init__()
        self.worker = worker
        self.new_positions = new_positions
        self.state = _GraphState({}, {})
        self.sources: list[Source] = []
        self.selected: str | None = None
        self.shown: dict[str, _Shown] = {}
        self.last_generation = 0
        self.windows: dict[str, mg.HostWindow] = {}
        self.panel_views: dict[str, OutputView] = {}
        self.panels_added = 0
        self.window_previewed = False
        self.profiled = False
        self.profiled_for_node_times = False
        self.playing = True
        self.fitted = False
        self.sequence = 0
        self.change_sequence = 0
        self.properties_scheduled = False

        self.graph_view = _GraphView()
        self.graph_view.connection_requested.connect(
            functools.partial(worker.post, worker.add_connection)
        )
        self.graph_view.output_clicked.connect(
            lambda identifier, output: self.show_source(
                _PREVIEW, self.output_source(identifier, output)
            )
        )
        self.graph_view.output_menu_requested.connect(self.output_menu)
        self.graph_view.delete_requested.connect(
            functools.partial(worker.post, worker.delete)
        )
        self.graph_view.node_selected.connect(self.select)
        self.graph_view.node_requested.connect(
            functools.partial(worker.post, worker.add_node)
        )
        self.graph_view.sink_changed.connect(
            lambda properties: self.set_view_properties(_BACKDROP, properties)
        )
        self.graph_view.input.message.connect(self.show_message)
        self.properties = QtProperties(self.schedule_properties, _property_pointer)
        self.profiler = ProfilerView()
        self.profiler.enabled_changed.connect(
            functools.partial(worker.post, worker.set_profiling)
        )
        self.profiler.enabled_changed.connect(self.profiling_changed)
        self.profiler.kind_changed.connect(
            functools.partial(worker.post, worker.set_report_kind)
        )
        self.profiler.interval_changed.connect(
            functools.partial(worker.post, worker.set_profiler_interval)
        )
        self.profiler.restart_requested.connect(
            functools.partial(worker.post, worker.restart_profiler_report)
        )
        self.memory = MemoryView()
        self.memory.node_clicked.connect(self.graph_view.select_node)
        self.preview = OutputView(
            "select what to preview above, or click an output", "Show nothing"
        )
        self.connect_view(_PREVIEW, self.preview)
        self.panel = OutputPanel()
        self.panel.hide()
        self.tabs = QtWidgets.QTabWidget()
        for page, name in (
            (self.properties.widget, "Node"),
            (self.profiler, "Times"),
            (self.memory, "Memory"),
        ):
            self.tabs.addTab(page, name)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.side = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.side.addWidget(self.preview)
        self.side.addWidget(self.tabs)
        self.side.setSizes([350, 550])
        self.split = QtWidgets.QSplitter()
        self.split.addWidget(self.graph_view)
        self.split.addWidget(self.panel)
        self.split.addWidget(self.side)
        self.split.setSizes([1000, 450, 600])
        self.setCentralWidget(self.split)
        self.add_menus()
        self.add_footer()
        self.node_times_action.setChecked(True)

        worker.nodes_changed.connect(self.refresh)
        worker.node_added.connect(self.node_added)
        worker.properties_recorded.connect(self.show_recording)
        worker.clocked.connect(self.show_clock)
        worker.reported.connect(self.show_report)
        worker.profiler_interval_changed.connect(self.profiler.set_interval)
        worker.captured.connect(self.show_capture)
        worker.window_shown.connect(self.window_shown)
        worker.playing_changed.connect(self.show_playing)
        worker.message.connect(self.show_message)
        QtCore.QTimer(
            self,
            interval=_PROPERTIES_INTERVAL // timedelta(milliseconds=1),
            timeout=self.request_properties,
        ).start()
        self.frame_timer = QtCore.QTimer(
            self,
            interval=_FRAME_INTERVAL // timedelta(milliseconds=1),
            timeout=self.show_frames,
            timerType=QtCore.Qt.TimerType.PreciseTimer,
        )
        self.set_path(path)

    def add_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        _add_action(
            file_menu, "&Open...", self.open_config, QtGui.QKeySequence.StandardKey.Open
        )
        _add_action(
            file_menu, "&Save...", self.save_config, QtGui.QKeySequence.StandardKey.Save
        )
        file_menu.addSeparator()
        _add_action(file_menu, "&Quit", self.close, QtGui.QKeySequence.StandardKey.Quit)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.setToolTipsVisible(True)
        _add_action(
            view_menu,
            "&Swap Preview and Graph",
            self.swap,
            "F2",
            "The preview large, the graph with its backdrop above the properties",
            checkable=True,
        )
        _add_action(
            view_menu, "Add &Panel", self.add_panel, tip="A view next to the graph"
        )
        _add_action(view_menu, "&Fit Graph", self.graph_view.fit)
        view_menu.addSeparator()
        self.backdrop_menu = view_menu.addMenu("&Backdrop")
        self.show_backdrop_sources()
        self.control_action = _add_action(
            view_menu,
            "&Control Backdrop Window",
            self.set_backdrop_control,
            "F3",
            "Send the input of the graph to the backdrop's window instead of editing",
            checkable=True,
        )
        view_menu.addSeparator()
        self.node_times_action = _add_action(
            view_menu,
            "Show Node &Times",
            self.show_node_times,
            "F4",
            "Profile and show the GPU time of each node above it, the hotter the "
            "redder",
            checkable=True,
        )

    def swap(self) -> None:
        split_sizes, side_sizes = self.split.sizes(), self.side.sizes()
        large, small = self.split.widget(0), self.side.widget(0)
        self.split.insertWidget(0, small)
        self.side.insertWidget(0, large)
        self.split.setSizes(split_sizes)
        self.side.setSizes(side_sizes)
        QtCore.QTimer.singleShot(0, self.graph_view.fit)

    def run_controls(self) -> QtWidgets.QWidget:
        controls = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(controls)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(2)

        def button(
            icon: str, tooltip: str, clicked: Callable[[], None]
        ) -> QtWidgets.QToolButton:
            tool = QtWidgets.QToolButton(
                icon=_painted_icon(icon), toolTip=tooltip, autoRaise=True
            )
            tool.setIconSize(QtCore.QSize(18, 18))
            tool.clicked.connect(clicked)
            layout.addWidget(tool)
            return tool

        self.play_button = button(
            "play",
            "",
            lambda: self.worker.post(self.worker.set_playing, not self.playing),
        )
        self.step_button = button(
            "step",
            "Step: runs one iteration",
            lambda: self.worker.post(self.worker.step),
        )
        button(
            "stop",
            "Stop: completes the iterations in flight and reconnects, which restarts the "
            "run at iteration 0",
            lambda: self.worker.post(self.worker.stop),
        )
        button(
            "reconnect",
            "Reconnect the graph",
            lambda: self.worker.post(self.worker.connect_graph),
        )
        button(
            "shaders",
            "Reload the shaders whose files changed",
            lambda: self.worker.post(self.worker.reload_shaders),
        )
        return controls

    def add_footer(self) -> None:
        footer = self.statusBar()
        footer.addWidget(self.run_controls())
        self.message_label = QtWidgets.QLabel()
        self.message_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred
        )
        footer.addWidget(self.message_label, 1)

        # Tabular digits
        stats = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(stats)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(0)
        font = QtGui.QFont(QtWidgets.QApplication.font())
        font.setFeature(QtGui.QFont.Tag("tnum"), 1)
        metrics = QtGui.QFontMetrics(font)

        def stat(caption: str, widest: str) -> QtWidgets.QLabel:
            if layout.count():
                layout.addSpacing(20)
            caption_label = QtWidgets.QLabel(caption)
            caption_label.setObjectName("footerCaption")
            value = QtWidgets.QLabel()
            value.setObjectName("footerValue")
            value.setFont(font)
            value.setAlignment(RIGHT_ALIGNED)
            value.setMinimumWidth(metrics.horizontalAdvance(widest))
            layout.addWidget(caption_label)
            layout.addWidget(value)
            return value

        self.iteration_label = stat("iteration", "8888888")
        self.time_label = stat("time", "8888.88 s")
        self.memory_label = stat("memory", WIDEST_SIZE)
        footer.addPermanentWidget(stats)
        self.show_playing(True)

    def show_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setToolTip(text)

    def show_playing(self, playing: bool) -> None:
        self.playing = playing
        self.play_button.setIcon(_painted_icon("pause" if playing else "play"))
        self.play_button.setToolTip(
            "Pause: holds the graph time" if playing else "Play"
        )
        self.step_button.setEnabled(not playing)

    def show_clock(self, iteration: int, graph_time: float) -> None:
        self.iteration_label.setText(str(iteration))
        self.time_label.setText(f"{graph_time:.2f} s")

    def show_report(self, report: dict) -> None:
        if "memory" in report:
            self.memory_label.setText(format_size(report["memory"]))
        if "profile" in report:
            if self.node_times_action.isChecked():
                self.graph_view.set_node_times(node_times(report["profile"]))
            self.profiler.add_sample(report["profile"])
        if "profile shown" in report:
            self.profiler.show_report(report["profile shown"])
        if "memory report" in report:
            self.memory.show_report(report["memory report"])

    def tab_changed(self) -> None:
        if self.tabs.currentWidget() is self.profiler:
            self.enable_profiling()
        self.post_reports()
        self.request_properties()

    def enable_profiling(self) -> None:
        if not self.profiled:
            self.profiled = True
            self.profiled_for_node_times = False
            self.profiler.set_enabled(True)

    def post_reports(self) -> None:
        times = (
            self.tabs.currentWidget() is self.profiler
            or self.node_times_action.isChecked()
        )
        memory = self.tabs.currentWidget() is self.memory
        self.worker.post(self.worker.set_reports, times, memory)

    def show_node_times(self, shown: bool) -> None:
        if shown:
            self.profiled_for_node_times = not self.profiler.enabled.isChecked()
            self.profiler.set_enabled(True)
        else:
            if self.profiled_for_node_times:
                self.profiler.set_enabled(False)
            self.graph_view.set_node_times({})
        self.post_reports()

    def profiling_changed(self, enabled: bool) -> None:
        if not enabled:
            self.node_times_action.setChecked(False)

    def refresh(self, state: _GraphState) -> None:
        if state == self.state and not self.new_positions:
            return
        self.state = state
        self.sources = _view_sources(state)
        for output_view in (self.preview, *self.panel_views.values()):
            output_view.set_sources(self.sources)
        self.show_backdrop_sources()
        self.graph_view.show_graph(
            state.nodes, self.new_positions, self.shown_outputs(), self.selected
        )
        self.new_positions = {}
        if not self.fitted:
            self.fitted = True
            QtCore.QTimer.singleShot(0, self.graph_view.fit)
        if self.selected not in state.nodes:
            self.select(None)
        for view, shown in list(self.shown.items()):
            if shown.source not in self.sources:
                self.hide_source(view)
        self.preview_first_window()

    def shown_outputs(self) -> set[tuple[str, str]]:
        return {
            (shown.source.identifier, shown.source.output)
            for shown in self.shown.values()
            if shown.source.kind == SourceKind.OUTPUT
        }

    def node_added(self, identifier: str, position: QtCore.QPointF) -> None:
        self.new_positions[identifier] = position
        self.select(identifier)

    def output_source(self, identifier: str, output: str) -> Source:
        nodes = self.state.nodes
        kind = (
            nodes[identifier].output_kind(output)
            if identifier in nodes
            else mg.ConnectorKind.OTHER
        )
        return Source(SourceKind.OUTPUT, identifier, output, kind)

    def output_menu(
        self, identifier: str, output: str, position: QtCore.QPoint
    ) -> None:
        source = self.output_source(identifier, output)
        showable = source.content in _SINK_TYPES
        menu = QtWidgets.QMenu(self)
        for view, label in ((_PREVIEW, "Preview"), (_BACKDROP, "Backdrop")):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.shows(view, source))
            action.setEnabled(showable and _can_show(view, source))
            action.toggled.connect(functools.partial(self.set_shown, view, source))
        panels = [view for view in self.panel_views if self.shows(view, source)]

        def set_panel_shown(shown: bool) -> None:
            if shown:
                self.add_panel(source=source)
            else:
                for view in panels:
                    self.remove_panel(view)

        action = menu.addAction("Panel")
        action.setCheckable(True)
        action.setChecked(bool(panels))
        action.setEnabled(showable)
        action.toggled.connect(set_panel_shown)
        menu.exec(position)
        menu.deleteLater()

    def shows(self, view: str, source: Source) -> bool:
        return view in self.shown and self.shown[view].source == source

    def set_shown(self, view: str, source: Source, shown: bool) -> None:
        if shown:
            self.show_source(view, source)
        else:
            self.hide_source(view)

    def show_source(self, view: str, source: Source | None) -> None:
        if source is None:
            self.hide_source(view)
            return
        if source.content not in _SINK_TYPES or not _can_show(view, source):
            self.show_message(f"{source.label()} cannot be shown there")
            return
        # A window or a sink of the graph has one consumer
        if source.kind in (SourceKind.WINDOW, SourceKind.SINK):
            for other, shown in list(self.shown.items()):
                if other != view and shown.source == source:
                    self.hide_source(other)
        self.release_window(view)
        self.last_generation += 1
        generation = self.last_generation
        self.shown[view] = _Shown(source, generation)
        if view == _BACKDROP:
            self.graph_view.show_capture(None)
            self.show_backdrop_sources()
            interval = timedelta(0)
        else:
            output_view = self.output_view(view)
            output_view.set_source(source)
            interval = output_view.capture_interval()
        self.graph_view.set_shown_outputs(self.shown_outputs())
        properties = self.view_of(view).sink_properties()
        self.worker.post(
            self.worker.show, view, generation, source, properties, interval
        )

    def hide_source(self, view: str) -> None:
        if view in self.shown:
            self.clear_view(view)
            self.graph_view.set_shown_outputs(self.shown_outputs())
            self.worker.post(self.worker.hide, view)

    def clear_view(self, view: str) -> None:
        """Clears the view in the window only."""
        del self.shown[view]
        self.release_window(view)
        if view == _BACKDROP:
            self.graph_view.show_capture(None)
            self.show_backdrop_sources()
        else:
            self.output_view(view).set_source(None)

    def show_backdrop_sources(self) -> None:
        shown = self.shown[_BACKDROP].source if _BACKDROP in self.shown else None
        set_source_actions(
            self.backdrop_menu,
            [source for source in self.sources if _can_show(_BACKDROP, source)],
            shown,
            lambda source: self.show_source(_BACKDROP, source),
        )

    def preview_first_window(self) -> None:
        if self.window_previewed:
            return
        if window := next(
            (source for source in self.sources if source.kind == SourceKind.WINDOW),
            None,
        ):
            self.window_previewed = True
            if _PREVIEW not in self.shown and all(
                shown.source != window for shown in self.shown.values()
            ):
                self.show_source(_PREVIEW, window)

    def output_view(self, view: str) -> OutputView:
        return self.preview if view == _PREVIEW else self.panel_views[view]

    def view_of(self, view: str) -> _GraphView | OutputView:
        return self.graph_view if view == _BACKDROP else self.output_view(view)

    def add_panel(self, *, source: Source | None = None) -> None:
        self.panels_added += 1
        view = f"panel {self.panels_added}"
        output_view = OutputView(
            "select what the panel shows above", "Remove the panel"
        )
        output_view.set_sources(self.sources)
        self.panel_views[view] = output_view
        self.connect_view(view, output_view)
        self.panel.add(output_view)
        self.panel.show()
        if source is not None:
            self.show_source(view, source)

    def remove_panel(self, view: str) -> None:
        self.hide_source(view)
        self.panel_views.pop(view).deleteLater()
        self.panel.setVisible(bool(self.panel_views))

    def connect_view(self, view: str, output_view: OutputView) -> None:
        def close() -> None:
            if view in self.panel_views:
                self.remove_panel(view)
            else:
                self.hide_source(view)

        output_view.closed.connect(close)
        output_view.source_requested.connect(
            lambda source: self.show_source(view, source)
        )
        output_view.sink_changed.connect(
            lambda properties: self.set_view_properties(view, properties)
        )
        output_view.interval_changed.connect(
            functools.partial(self.worker.post, self.worker.set_capture_interval, view)
        )
        output_view.input.message.connect(self.show_message)

    def set_view_properties(self, view: str, properties: dict) -> None:
        if window := self.windows.get(view):
            window.set_extent(
                *properties["fit"], self.view_of(view).devicePixelRatioF()
            )
        else:
            self.worker.post(self.worker.set_view_properties, view, properties)

    def show_capture(self, view: str, generation: int, capture: mg.Capture) -> None:
        shown = self.shown.get(view)
        if shown is None or shown.generation != generation:
            return
        self.view_of(view).show_capture(capture)
        self.worker.post(self.worker.capture_shown, view, generation)

    def window_shown(
        self, view: str, generation: int, window: mg.HostWindow | None
    ) -> None:
        shown = self.shown.get(view)
        if shown is None or shown.generation != generation:
            return
        self.release_window(view)
        widget = self.view_of(view)
        if window is None:
            widget.show_capture(None)
            return
        # Drops a frame of an earlier show
        window.get_new_frame()
        self.windows[view] = window
        if view != _BACKDROP or self.control_action.isChecked():
            widget.input.set_window(window)
        self.set_view_properties(view, widget.sink_properties())
        self.frame_timer.start()

    def release_window(self, view: str) -> None:
        """The window presents nothing until a view shows it again."""
        if window := self.windows.pop(view, None):
            window.set_extent(0, 0)
            window.get_new_frame()
            self.view_of(view).input.set_window(None)
        if not self.windows:
            self.frame_timer.stop()

    def show_frames(self) -> None:
        for view, window in self.windows.items():
            widget = self.view_of(view)
            if frame := window.get_new_frame():
                widget.show_capture(frame)
            widget.input.update()

    def set_backdrop_control(self, control: bool) -> None:
        self.graph_view.input.set_window(
            self.windows.get(_BACKDROP) if control else None
        )
        if control:
            self.graph_view.setFocus()

    def select(self, identifier: str | None) -> None:
        self.memory.set_selected(identifier)
        self.profiler.set_selected(identifier)
        if identifier != self.selected:
            # The click that selects commits the edit in progress to the previous node
            if self.properties.changes:
                self.request_properties()
            self.selected = identifier
            self.properties.clear()
            self.request_properties()

    def schedule_properties(self) -> None:
        if not self.properties_scheduled:
            self.properties_scheduled = True
            QtCore.QTimer.singleShot(0, self.request_properties)

    def request_properties(self) -> None:
        self.properties_scheduled = False
        shown = self.tabs.currentWidget() is self.properties.widget
        if not self.selected or not shown or QtWidgets.QApplication.activeModalWidget():
            return
        self.sequence += 1
        recorder = self.properties.recorder()
        if recorder.changes:
            self.change_sequence = self.sequence
        sequence, identifier = self.sequence, self.selected
        self.worker.post(self.worker.request_properties, sequence, identifier, recorder)

    def show_recording(self, sequence: int, identifier: str, items: list) -> None:
        if identifier == self.selected:
            self.properties.show(items, consumed=sequence >= self.change_sequence)

    def open_config(self) -> None:
        directory = str(self.path.parent) if self.path else str(scene.EXAMPLES)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open graph", directory, "Graph config (*.json)"
        )
        if not path:
            return
        try:
            description = mg.config.load(path)
        except Exception as error:
            self.show_message(str(error))
            return
        self.set_path(Path(path))
        for view in list(self.shown):
            self.clear_view(view)
        self.window_previewed = False
        self.graph_view.clear_graph()
        self.state = _GraphState({}, {})
        self.fitted = False
        self.new_positions = _stored_positions(description)
        self.worker.post(self.worker.load, description)

    def save_config(self) -> None:
        stem = self.path.stem if self.path else "graph"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save graph",
            str(Path.cwd() / f"{stem}.stored.json"),
            "Graph config (*.json)",
        )
        if path:
            positions = self.graph_view.positions()
            self.worker.post(self.worker.save, Path(path), positions)

    def set_path(self, path: Path | None) -> None:
        self.path = path
        self.setWindowTitle(f"{path} - merian" if path else "merian")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "scene",
        type=Path,
        nargs="?",
        help="a scene or a graph config (.json); without, an empty graph",
    )
    _, config = scene.parse(parser, argv)
    if config:
        path, description = config.path, config.description()
        new_positions = _stored_positions(description)
    else:
        path, description, new_positions = None, None, {}

    application = QtWidgets.QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    application.setPalette(theme_palette())
    application.setStyleSheet(theme_stylesheet())
    # Queued, as a quit() before exec() is lost
    signal.signal(
        signal.SIGINT, lambda *_: QtCore.QTimer.singleShot(0, application.quit)
    )
    with mg.Graph(
        scene.context((mg.HOST_WINDOWS_EXTENSION,)), description, strict=False
    ) as graph:
        worker = _GraphWorker(graph)
        window = _Window(worker, path, new_positions)
        window.resize(1600, 900)
        window.show()
        worker.start()
        try:
            return application.exec()
        finally:
            worker.quit()


if __name__ == "__main__":
    sys.exit(main())
