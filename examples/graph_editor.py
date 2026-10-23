#!/usr/bin/env python3
"""
A graph editor: the nodes and their connections, the properties of the selected node,
times, memory, and what the graph shows: outputs, and the windows of its Window nodes,
which take the input of their view.

  graph_editor.py [scene or graph.json] [graph options]

Right-click adds a node. Drag from an output to an input to connect them; pull a
connection off its input to move or remove it. Click a node for its properties and its
time for the profiler. Click an output to preview it, left of the graph and properties;
right-click it to show it as the backdrop of the graph or in a panel next to the graph,
or where its memory is. The header of each view, and View > Backdrop for the backdrop,
select what it shows. Delete removes the selected
nodes and connections. Hollow inputs are optional; grey connectors and dashed
connections were connected but are not described by their node now. Positions are
stored in the saved config.

The graph navigates like Blender's node editor: the middle button pans, with Ctrl it
zooms, the wheel and a pinch zoom, Shift+wheel and two fingers pan, Home shows all nodes.
Right-click a node for its menu. Shift+A adds a
node, G moves the selected ones, X deletes, Ctrl+right-drag cuts connections, Shift+click
adds to the selection.

Space plays and pauses, Right steps. A graph's view sends the keys to the graph; there,
Shift+Alt+Space plays and pauses, Shift+Alt+Right steps and Shift+Alt+Backspace gives the
mouse back, in the editor too; the Preferences tab sets their modifier, and the log
level. The Log tab shows merian's log. The preferences are stored for the user, not in
the config.
"""

import argparse
import functools
import itertools
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
import merian
import merian.graph as mg
from gui.log_view import LogView
from gui.output_view import (
    CaptureImage,
    OutputView,
    PanelColumn,
    Source,
    SourceKind,
    fit_properties,
    resize_timer,
    save_capture,
    set_source_actions,
)
from gui.preferences_view import PreferencesView
from gui.qt_properties import QtProperties, Recorder
from gui.report_views import (
    WIDEST_SIZE,
    MemoryView,
    ProfilerView,
    TimeFormat,
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
_SETTINGS = ("merian", "graph-editor")

_CLOCK_INTERVAL = timedelta(milliseconds=33)
_BUSY_INTERVAL = timedelta(milliseconds=80)
_SPINNER = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
_CONNECTED = "/graph/connect"
# The sequence of the properties published with the graph
_PUBLISHED = 0
_SAVE_FILTERS = {
    mg.ConnectorKind.IMAGE: ("png", "Images (*.png *.jpg *.bmp *.tif)"),
    mg.ConnectorKind.BUFFER: ("bin", "Raw data (*.bin)"),
}
_FRAME_INTERVAL = timedelta(milliseconds=4)
_MEMORY_INTERVAL = timedelta(seconds=1)
_PROPERTIES_INTERVAL = timedelta(milliseconds=100)
_REPORT_INTERVAL = timedelta(milliseconds=100)
_RUN_SLICE = timedelta(milliseconds=15)

_PLAYBACK_KEYS = (QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_Right)
_RELEASE_MOUSE_KEY = QtCore.Qt.Key.Key_Backspace
_PLAYBACK_KEY_USERS = (
    QtWidgets.QAbstractButton,
    QtWidgets.QAbstractItemView,
    QtWidgets.QAbstractSlider,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QComboBox,
    QtWidgets.QLineEdit,
    QtWidgets.QPlainTextEdit,
)
_STOP_TIP = (
    "Stop: completes the iterations in flight and reconnects, which restarts the run at "
    "iteration 0"
)

_BACKDROP_SHADE = QtGui.QColor(THEME["canvas"])
_BACKDROP_SHADE.setAlpha(120)
_VEIL_COLOR = QtGui.QColor(THEME["bg"])
_VEIL_COLOR.setAlpha(170)
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
_MAX_SCALE = 4.0
_MIN_FIT_SCALE = 0.8
_MIN_SCALE = 0.1
_NODE_SPACING = 30
_ROW_HEIGHT = 20
_SHADOW = ((5, 22), (2, 34))
_SHADOW_OFFSET = 3
# The center of the first connector row
_TITLE_HEIGHT = 44
_WORLD = QtCore.QRectF(-1e5, -1e5, 2e5, 2e5)
_WIDEST_TIME = "88.88 µs"


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
class _Save:
    path: Path
    first: int
    end: int | None = None
    properties: dict | None = None


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
    save: _Save | None = None

    def whole(self, capture: mg.Capture) -> bool:
        save = self.save
        return (
            save is not None
            and capture.run_iteration >= save.first
            and (save.end is None or capture.run_iteration < save.end)
        )


class _GraphWorker(QtCore.QObject):
    """Makes every graph call on its own thread; commands run between iterations."""

    nodes_changed = QtCore.Signal(object)
    node_added = QtCore.Signal(str, object)
    # sequence, identifier or None for the graph, items
    properties_recorded = QtCore.Signal(int, object, object)
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
    # What is loading, empty when done
    loading = QtCore.Signal(str)
    # path, capture
    save_captured = QtCore.Signal(object, object)

    def __init__(
        self,
        open_context: Callable[[], merian.Context],
        description: mg.Description | None,
    ):
        super().__init__()
        self.open_context, self.description = open_context, description
        self.context: merian.Context | None = None
        self.graph: mg.Graph | None = None
        self.commands: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self.runner = threading.Thread(target=self.loop, name="graph", daemon=True)
        self.playing = True
        self.steps = 0
        self.paused_time: float | None = None
        self.closing = threading.Event()
        # By whether for the graph: the expanded children and tabs last requested
        self.expansions: dict[bool, tuple[dict, dict]] = {}
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
        self.runner.start()

    def quit(self) -> None:
        self.closing.set()
        # Wakes the loop
        self.post(lambda: None)
        self.runner.join()

    def open_graph(self) -> None:
        self.loading.emit("Loading merian")
        self.context = self.open_context()
        self.loading.emit("Loading graph")
        try:
            self.graph = mg.Graph(self.context, self.description, strict=False)
        except Exception:
            self.graph = mg.Graph(self.context)
            raise
        finally:
            self.loading.emit("")
            if self.graph is not None and not self.closing.is_set():
                self.connect_graph()

    @property
    def running(self) -> bool:
        return self.playing or self.steps > 0

    def loop(self) -> None:
        try:
            self.attempt(self.open_graph)
            if self.graph is None:
                self.loading.emit("")
                self.closing.wait()
                return
            self.serve()
        finally:
            if self.graph is not None:
                self.graph.close()
            self.graph = self.context = None

    def serve(self) -> None:
        while True:
            self.run_commands(block=not self.running)
            if self.closing.is_set():
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
        if not self.playing:
            limits = {"iterations": 1}
        elif self.paused_time is not None:
            limits = {"iterations": 1, "time": self.paused_time}
            self.paused_time = None
        else:
            limits = {"iterations": None, "duration": _RUN_SLICE}
        self.steps = max(self.steps - 1, 0)
        try:
            result = self.graph.run(**limits, until_event=_CONNECTED)
            if not self.playing:
                self.paused_time = self.graph.time
            if result.stop_reason == mg.StopReason.EVENT:
                self.restart_saves()
                self.publish_state()
        except Exception:
            self.set_playing(False)
            raise

    def take_captures(self) -> None:
        """Publishes the latest captures, to be confirmed with capture_shown."""
        for view, view_sink in self.view_sinks.items():
            if not (captures := view_sink.sink.get_all()):
                continue
            view_sink.requested = False
            if (save := view_sink.save) and save.end is None:
                if whole := next(filter(view_sink.whole, captures), None):
                    self.complete_save(view_sink, whole)
            shown = [capture for capture in captures if not view_sink.whole(capture)]
            if (save := view_sink.save) and save.end is not None:
                if any(capture.run_iteration >= save.end for capture in captures):
                    view_sink.save = None
            if shown:
                view_sink.unshown += 1
                self.captured.emit(view, view_sink.generation, shown[-1])

    def save_capture(self, view: str, path: Path, properties: dict) -> None:
        view_sink = self.view_sinks.get(view)
        if view_sink is None:
            raise RuntimeError("the view shows nothing to save")
        if view_sink.save and view_sink.save.end is None:
            self.message.emit(
                f"not saved {view_sink.save.path}, a newer save replaced it"
            )
        view_sink.save = _Save(path, self.graph.run_iteration)
        if view_sink.owned:
            view_sink.save.properties = properties
            whole = (
                {"fit": [0, 0]}
                if view_sink.content == mg.ConnectorKind.IMAGE
                else {"offset": 0, "size": 0}
            )
            self.graph.nodes[view_sink.identifier].set_properties(whole)
        view_sink.sink.capture()
        if not self.playing:
            self.steps += 1

    def complete_save(self, view_sink: _ViewSink, capture: mg.Capture) -> None:
        save = view_sink.save
        save.end = self.graph.run_iteration
        if save.properties is not None:
            self.graph.nodes[view_sink.identifier].set_properties(save.properties)
        self.save_captured.emit(save.path, capture)

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
        self.restart_saves()
        self.show_profiler_interval()
        self.publish_state()

    def restart_saves(self) -> None:
        for view_sink in self.view_sinks.values():
            if (save := view_sink.save) and save.end is None:
                save.first = self.graph.run_iteration
                view_sink.sink.capture()
            elif save:
                view_sink.save = None

    def publish_state(self) -> None:
        nodes = self.graph.nodes
        self.view_sinks = {
            view: view_sink
            for view, view_sink in self.view_sinks.items()
            if view_sink.identifier in nodes
        }
        state = _read_state(self.graph, self.owned_sinks())
        self.nodes_changed.emit(state)
        for identifier in (None, *state.nodes):
            self.attempt(
                functools.partial(
                    self.record,
                    _PUBLISHED,
                    identifier,
                    Recorder({}, *self.expansions.get(identifier is None, ({}, {}))),
                )
            )
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

    def move_connection(
        self,
        old: tuple[str, str, str],
        src: str,
        src_output: str,
        dst: str,
        dst_input: str,
    ) -> None:
        def move(graph: mg.Graph) -> None:
            graph.add_connection(src, dst, src_output, dst_input)
            graph.remove_connection(*old)

        self.edit(move)

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
        view_sink.save = None
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
            if view_sink.save and view_sink.save.end is None:
                view_sink.save.properties = properties
            else:
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
        self, sequence: int, identifier: str | None, recorder: Recorder
    ) -> None:
        """Records the properties of a node, of the graph for identifier None."""
        self.expansions[identifier is None] = (recorder.expanded, recorder.tabs)
        if self.properties_request:
            _, pending_identifier, pending = self.properties_request
            if pending_identifier == identifier:
                recorder.changes = pending.changes | recorder.changes
            else:
                self.record_properties(*self.properties_request)
        self.properties_request = (sequence, identifier, recorder)

    def record(self, sequence: int, identifier: str | None, recorder: Recorder) -> None:
        nodes = self.graph.nodes
        if identifier is None:
            record = self.graph.properties
        elif identifier in nodes:
            record = nodes[identifier].properties
        else:
            return
        try:
            record(recorder)
        finally:
            self.properties_recorded.emit(sequence, identifier, recorder.items)

    def record_properties(
        self, sequence: int, identifier: str | None, recorder: Recorder
    ) -> None:
        changed = bool(recorder.changes)
        try:
            self.record(sequence, identifier, recorder)
        finally:
            if self.graph.needs_reconnect:
                self.connect_graph()
            elif identifier is None and changed:
                self.show_profiler_interval()
            if identifier is None and changed and not self.playing:
                source = self.graph.get_properties()["graph_properties"]["time source"]
                if source == "Time":
                    self.paused_time = None

    def show_profiler_interval(self) -> None:
        interval = self.graph.get_properties()["profiler"]["report interval"]
        if interval != self.profiler_interval:
            self.profiler_interval = interval
            self.profiler_interval_changed.emit(interval)


def _menu_on_release() -> bool:
    trigger = QtGui.QGuiApplication.styleHints().contextMenuTrigger()
    return trigger == QtCore.Qt.ContextMenuTrigger.Release


@functools.cache
def _bold_font() -> QtGui.QFont:
    font = QtGui.QFont(QtWidgets.QApplication.font())
    font.setBold(True)
    return font


@functools.cache
def _tabular_font() -> QtGui.QFont:
    font = QtGui.QFont(QtWidgets.QApplication.font())
    font.setFeature(QtGui.QFont.Tag("tnum"), 1)
    return font


@functools.cache
def _tabular_metrics() -> QtGui.QFontMetricsF:
    return QtGui.QFontMetricsF(_tabular_font())


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
        # GPU time and share of the total
        self.time: str | None = None
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

    def set_time(self, time: str | None, share: float) -> None:
        if (time, share) != (self.time, self.share):
            self.time, self.share = time, share
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

        if (chip := self.chip_rect()) is not None:
            heat = min(self.share * 2, 1.0)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(_mixed(THEME["surface_hi"], THEME["red"], heat))
            painter.drawRoundedRect(chip, _CHIP_HEIGHT / 2, _CHIP_HEIGHT / 2)
            painter.setPen(QtGui.QColor(THEME["bg" if heat > 0.5 else "text"]))
            painter.setFont(_tabular_font())
            painter.drawText(chip, QtCore.Qt.AlignmentFlag.AlignCenter, self.time)

    def chip_rect(self) -> QtCore.QRectF | None:
        if self.time is None:
            return None
        width = (
            max(
                _tabular_metrics().horizontalAdvance(text)
                for text in (self.time, _WIDEST_TIME)
            )
            + 14
        )
        rect = self.rect()
        return QtCore.QRectF(
            rect.right() - width,
            rect.top() - _CHIP_HEIGHT - _CHIP_GAP,
            width,
            _CHIP_HEIGHT,
        )

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

    def connection(self) -> tuple[str, str, str]:
        """Returns (src, dst, dst_input)."""
        return (
            self.source.node.identifier,
            self.target.node.identifier,
            self.target.name,
        )

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


def _segments(path: QtGui.QPainterPath) -> list[QtCore.QLineF]:
    return [
        QtCore.QLineF(start, end)
        for polygon in path.toSubpathPolygons()
        for start, end in itertools.pairwise(polygon)
    ]


def _shift_as_control(event: QtGui.QMouseEvent) -> QtGui.QMouseEvent:
    modifiers = QtCore.Qt.KeyboardModifier
    if not event.modifiers() & modifiers.ShiftModifier:
        return event
    return QtGui.QMouseEvent(
        event.type(),
        event.position(),
        event.globalPosition(),
        event.button(),
        event.buttons(),
        event.modifiers() & ~modifiers.ShiftModifier | modifiers.ControlModifier,
    )


class _GraphView(QtWidgets.QGraphicsView):
    # src, src_output, dst, dst_input
    connection_requested = QtCore.Signal(str, str, str, str)
    # (src, dst, dst_input) it leaves, src, src_output, dst, dst_input
    connection_moved = QtCore.Signal(object, str, str, str, str)
    # node, output
    output_clicked = QtCore.Signal(str, str)
    # node, output, global position
    output_menu_requested = QtCore.Signal(str, str, object)
    # node, global position
    node_menu_requested = QtCore.Signal(str, object)
    # identifiers, [(src, dst, dst_input)]
    delete_requested = QtCore.Signal(list, list)
    # identifier or None
    node_selected = QtCore.Signal(object)
    node_clicked = QtCore.Signal(str)
    time_clicked = QtCore.Signal(str)
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
        self.detached: _Edge | None = None
        self.drag_start = QtCore.QPointF()
        # The node a left press landed on, and where it was
        self.pressed: tuple[_NodeItem, QtCore.QPointF] | None = None
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QtGui.QColor(THEME["canvas"]))
        self.setDragMode(self.DragMode.RubberBandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.NoAnchor)
        self.setSceneRect(_WORLD)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewport().setMouseTracking(True)
        self.navigation: tuple[QtCore.QPointF, QtCore.QPointF] | None = None
        self.zooming = False
        self.grabbed: dict[_NodeItem, QtCore.QPointF] = {}
        self.grab_start = QtCore.QPointF()
        self.cut_line: QtWidgets.QGraphicsPathItem | None = None
        self.cut_points: list[QtCore.QPointF] = []
        self.menu_suppressed = False
        self.scene().selectionChanged.connect(self.emit_selection)
        self.node_times: dict[str, float] = {}
        self.time_formats: defaultdict[str, TimeFormat] = defaultdict(TimeFormat)
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
        self.end_gestures()
        graphics.clear()
        self.dragged = self.drag_line = self.detached = self.pressed = None
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

    def fit(self, min_scale: float = _MIN_FIT_SCALE) -> None:
        bounds = self.scene().itemsBoundingRect()
        self.resetTransform()
        if bounds.isEmpty():
            self.centerOn(0, 0)
            return
        self.fitInView(bounds, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        scale = min(max(self.transform().m11(), min_scale), 1.0)
        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(bounds.center())

    def zoom(self, factor: float, position: QtCore.QPointF) -> None:
        anchor = self.mapToScene(position.toPoint())
        current = self.transform().m11()
        factor = min(max(current * factor, _MIN_SCALE), _MAX_SCALE) / current
        self.scale(factor, factor)
        self.pan(position - QtCore.QPointF(self.mapFromScene(anchor)))

    def pan(self, delta: QtCore.QPointF) -> None:
        for bar, offset in (
            (self.horizontalScrollBar(), delta.x()),
            (self.verticalScrollBar(), delta.y()),
        ):
            bar.setValue(bar.value() - round(offset))

    def end_gestures(self) -> None:
        if self.cut_line:
            self.menu_suppressed = _menu_on_release()
        if self.grabbed or self.cut_line:
            self.grabbed, self.cut_line, self.cut_points = {}, None, []
            self.viewport().unsetCursor()

    def clear_graph(self) -> None:
        self.end_gestures()
        self.dragged = self.drag_line = self.detached = self.pressed = None
        self.scene().clear()
        self.items_by_identifier = {}

    def set_node_times(self, times: dict[str, float]) -> None:
        self.node_times = times
        for identifier in self.time_formats.keys() - times.keys():
            del self.time_formats[identifier]
        total = sum(times.values()) or 1.0
        for identifier, item in self.items_by_identifier.items():
            if (milliseconds := times.get(identifier)) is None:
                item.set_time(None, 0.0)
            else:
                item.set_time(
                    self.time_formats[identifier](milliseconds), milliseconds / total
                )

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
        if self.menu_suppressed or self.cut_line:
            self.menu_suppressed = False
            return
        connector = self.connector_at(QtCore.QPointF(event.pos()))
        if connector and connector.is_output:
            self.output_menu_requested.emit(
                connector.node.identifier, connector.name, event.globalPos()
            )
            return
        item = self.itemAt(event.pos())
        while item is not None and not isinstance(item, _NodeItem):
            item = item.parentItem()
        if item is not None:
            self.node_menu_requested.emit(item.identifier, event.globalPos())
            return
        self.show_picker(event.pos(), event.globalPos())

    def show_picker(
        self, position: QtCore.QPoint, global_position: QtCore.QPoint
    ) -> None:
        self.picker_position = self.mapToScene(position)
        self.picker.exec(global_position)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        buttons, modifiers = QtCore.Qt.MouseButton, QtCore.Qt.KeyboardModifier
        if self.grabbed:
            self.menu_suppressed = event.button() == buttons.RightButton
            self.end_grab(event.button() == buttons.LeftButton)
            return
        if self.drag_line:
            return
        if event.button() == buttons.MiddleButton:
            self.zooming = bool(event.modifiers() & modifiers.ControlModifier)
            position = event.position()
            self.navigation = (position, position)
            self.viewport().setCursor(
                QtCore.Qt.CursorShape.SizeVerCursor
                if self.zooming
                else QtCore.Qt.CursorShape.ClosedHandCursor
            )
            return
        if event.button() == buttons.RightButton and (
            event.modifiers() & modifiers.ControlModifier
        ):
            self.cut_points = [self.mapToScene(event.position().toPoint())]
            pen = QtGui.QPen(
                QtGui.QColor(THEME["red"]), 1.5, QtCore.Qt.PenStyle.DashLine
            )
            pen.setCosmetic(True)
            self.cut_line = self.scene().addPath(QtGui.QPainterPath(), pen)
            return
        plain = event.modifiers() == modifiers.NoModifier
        event = _shift_as_control(event)
        if event.button() != buttons.LeftButton:
            super().mousePressEvent(event)
            return
        scene_position = self.mapToScene(event.position().toPoint())
        if node := self.chip_at(scene_position):
            self.select_node(node.identifier)
            self.time_clicked.emit(node.identifier)
            return
        connector = self.connector_at(event.position())
        if connector and not connector.is_output and plain:
            self.detached = next(
                (edge for edge in connector.node.edges if edge.target is connector),
                None,
            )
            if self.detached:
                self.detached.hide()
                connector = self.detached.source
        if connector and connector.is_output:
            self.dragged = connector
            self.drag_start = event.position()
            line = QtCore.QLineF(connector.scenePos(), scene_position)
            pen = QtGui.QPen(QtGui.QColor(THEME["text_dim"]), 2)
            self.drag_line = self.scene().addLine(line, pen)
            return
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, _NodeItem):
            self.pressed = (item, item.pos())
        super().mousePressEvent(event)

    def chip_at(self, position: QtCore.QPointF) -> _NodeItem | None:
        covering = self.scene().itemAt(position, self.transform())
        if covering is not None and not isinstance(covering, _Edge):
            return None
        for item in self.items_by_identifier.values():
            chip = item.chip_rect()
            if chip is not None and item.mapRectToScene(chip).contains(position):
                return item
        return None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        position = event.position()
        if self.navigation:
            last, pressed = self.navigation
            self.navigation = (position, pressed)
            if self.zooming:
                self.zoom(1.005 ** (last.y() - position.y()), pressed)
            else:
                self.pan(position - last)
            return
        if self.grabbed:
            offset = self.mapToScene(position.toPoint()) - self.grab_start
            for item, start in self.grabbed.items():
                item.setPos(start + offset)
            return
        if self.cut_line:
            self.cut_points.append(self.mapToScene(position.toPoint()))
            path = QtGui.QPainterPath()
            path.addPolygon(QtGui.QPolygonF(self.cut_points))
            self.cut_line.setPath(path)
            return
        if self.drag_line:
            end = self.mapToScene(event.position().toPoint())
            self.drag_line.setLine(QtCore.QLineF(self.dragged.scenePos(), end))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        buttons = QtCore.Qt.MouseButton
        if self.navigation and event.button() == buttons.MiddleButton:
            self.navigation = None
            self.viewport().unsetCursor()
            return
        if self.cut_line and event.button() == buttons.RightButton:
            self.cut(self.cut_line.path())
            self.scene().removeItem(self.cut_line)
            self.cut_line = None
            self.menu_suppressed = _menu_on_release()
            return
        if not self.drag_line:
            pressed, self.pressed = self.pressed, None
            super().mouseReleaseEvent(_shift_as_control(event))
            if pressed and event.button() == buttons.LeftButton:
                node, position = pressed
                if node.pos() == position and node.isSelected():
                    self.node_clicked.emit(node.identifier)
            return
        if event.button() != buttons.LeftButton:
            return
        self.scene().removeItem(self.drag_line)
        source, self.dragged, self.drag_line = self.dragged, None, None
        detached, self.detached = self.detached, None
        target = self.connector_at(event.position())
        distance = (event.position() - self.drag_start).manhattanLength()
        if detached and (
            target is detached.target
            or (
                target is None and distance < QtWidgets.QApplication.startDragDistance()
            )
        ):
            detached.show()
        elif target is source and not detached:
            self.output_clicked.emit(source.node.identifier, source.name)
        elif target and not target.is_output:
            new = (
                source.node.identifier,
                source.name,
                target.node.identifier,
                target.name,
            )
            if detached:
                self.connection_moved.emit(detached.connection(), *new)
            else:
                self.connection_requested.emit(*new)
        elif detached:
            self.delete_requested.emit([], [detached.connection()])

    def cut(self, line: QtGui.QPainterPath) -> None:
        cut = _segments(line)
        connections = [
            item.connection()
            for item in self.scene().items(
                line.boundingRect().adjusted(-1, -1, 1, 1),
                QtCore.Qt.ItemSelectionMode.IntersectsItemBoundingRect,
            )
            if isinstance(item, _Edge)
            and any(
                segment.intersects(other)[0]
                == QtCore.QLineF.IntersectionType.BoundedIntersection
                for segment in _segments(item.path())
                for other in cut
            )
        ]
        if connections:
            self.delete_requested.emit([], connections)

    def start_grab(self) -> None:
        nodes = [
            item for item in self.scene().selectedItems() if isinstance(item, _NodeItem)
        ]
        if not nodes:
            return
        self.grab_start = self.mapToScene(
            self.viewport().mapFromGlobal(QtGui.QCursor.pos())
        )
        self.grabbed = {item: item.pos() for item in nodes}
        self.viewport().setCursor(QtCore.Qt.CursorShape.SizeAllCursor)

    def end_grab(self, confirmed: bool) -> None:
        if not confirmed:
            for item, start in self.grabbed.items():
                item.setPos(start)
        self.grabbed = {}
        self.viewport().unsetCursor()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        keys, modifiers = QtCore.Qt.Key, QtCore.Qt.KeyboardModifier
        key, plain = event.key(), event.modifiers() == modifiers.NoModifier
        if self.grabbed:
            if key in (keys.Key_Escape, keys.Key_Return, keys.Key_Enter):
                self.end_grab(key != keys.Key_Escape)
        elif key == keys.Key_Home:
            self.fit(_MIN_SCALE)
        elif key == keys.Key_A and event.modifiers() == modifiers.ShiftModifier:
            cursor = QtGui.QCursor.pos()
            self.show_picker(self.viewport().mapFromGlobal(cursor), cursor)
        elif key == keys.Key_G and plain:
            self.start_grab()
        elif key == keys.Key_Delete or (key == keys.Key_X and plain):
            self.delete_selected()
        else:
            super().keyPressEvent(event)

    def delete_selected(self) -> None:
        identifiers, connections = [], []
        for item in self.scene().selectedItems():
            if isinstance(item, _NodeItem):
                identifiers.append(item.identifier)
            elif isinstance(item, _Edge):
                connections.append(item.connection())
        self.delete_requested.emit(identifiers, connections)

    def event(self, event: QtCore.QEvent) -> bool:
        if (
            event.type() == QtCore.QEvent.Type.NativeGesture
            and event.gestureType() == QtCore.Qt.NativeGestureType.ZoomNativeGesture
        ):
            self.zoom(1.0 + event.value(), event.position())
            return True
        return super().event(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        modifiers = QtCore.Qt.KeyboardModifier
        delta = event.angleDelta()
        steps = delta.y() or delta.x()
        # Windows turns a pinch into Ctrl and the wheel
        if event.modifiers() & modifiers.ControlModifier:
            self.zoom(1.0015**steps, event.position())
        elif _from_touchpad(event):
            pixels = event.pixelDelta()
            self.pan(
                QtCore.QPointF(pixels)
                if not pixels.isNull()
                else QtCore.QPointF(delta) / 2
            )
        elif not delta.y() or event.modifiers() & modifiers.ShiftModifier:
            self.pan(QtCore.QPointF(steps / 2, 0))
        else:
            self.zoom(1.0015**steps, event.position())


def _from_touchpad(event: QtGui.QWheelEvent) -> bool:
    device = event.pointingDevice()
    if device is not None and device.type() == QtGui.QInputDevice.DeviceType.TouchPad:
        return True
    if event.phase() != QtCore.Qt.ScrollPhase.NoScrollPhase:
        return True
    # Mouse wheels turn in notches of 120
    delta = event.angleDelta()
    return not event.pixelDelta().isNull() or bool(delta.x() % 120 or delta.y() % 120)


def _pointer(path: list[str]) -> str:
    return "".join("/" + name.replace("~", "~0").replace("/", "~1") for name in path)


def _key_names(key: tuple) -> list[str] | None:
    if any(occurrence for _, _, occurrence in key):
        return None
    return [name for _, name, _ in key]


def _property_pointer(key: tuple) -> str | None:
    names = _key_names(key)
    if names is None:
        return None
    if len(names) == 2 and names[1] in ("enabled", "linearization order"):
        return _pointer(["nodes", *names])
    if len(names) > 3 and names[1:3] == ["sections", "properties"]:
        return _pointer(["nodes", names[0], "properties", *names[3:]])
    return None


def _graph_property_pointer(key: tuple) -> str | None:
    names = _key_names(key)
    if names and len(names) == 2 and names[0] == "graph_properties":
        return _pointer(names)
    return None


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


def _keys(action: QtGui.QAction) -> str:
    """The shortcuts of an action: in the editor, then in a graph's view."""
    editor, view = (
        shortcut.toString(QtGui.QKeySequence.SequenceFormat.NativeText)
        for shortcut in action.shortcuts()
    )
    return f"{editor}, in a graph's view {view}"


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


class _Veil(QtWidgets.QWidget):
    def __init__(self, covered: QtWidgets.QWidget):
        super().__init__(covered)
        covered.installEventFilter(self)
        self.setGeometry(covered.rect())

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Resize:
            self.setGeometry(watched.rect())
        return False

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        QtGui.QPainter(self).fillRect(self.rect(), _VEIL_COLOR)


class _Shown(NamedTuple):
    source: Source
    generation: int


class _Window(QtWidgets.QMainWindow):
    def __init__(
        self,
        worker: _GraphWorker,
        path: Path | None,
        new_positions: dict[str, QtCore.QPointF],
        log_sink: merian.LogSink,
        preferences: PreferencesView,
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
        self.node_times_resumed = False
        self.playing = True
        self.fitted = False
        self.sequence = 0
        self.change_sequences: dict[str | None, int] = {}
        self.properties_scheduled = False
        self.loading = ""
        self.held: list[QtGui.QAction] = []
        self.spin = 0
        self.recordings: dict[str | None, list] = {}
        self.view_shortcuts: set[int] = set()

        self.graph_view = _GraphView()
        self.graph_view.connection_requested.connect(
            functools.partial(worker.post, worker.add_connection)
        )
        self.graph_view.connection_moved.connect(
            functools.partial(worker.post, worker.move_connection)
        )
        self.graph_view.output_clicked.connect(
            lambda identifier, output: self.show_source(
                _PREVIEW, self.output_source(identifier, output)
            )
        )
        self.graph_view.output_menu_requested.connect(self.output_menu)
        self.graph_view.node_menu_requested.connect(self.node_menu)
        self.graph_view.delete_requested.connect(
            functools.partial(worker.post, worker.delete)
        )
        self.graph_view.node_selected.connect(self.select)
        self.graph_view.node_clicked.connect(
            lambda: self.tabs.setCurrentWidget(self.properties.widget)
        )
        self.graph_view.time_clicked.connect(self.show_node_time)
        self.graph_view.node_requested.connect(
            functools.partial(worker.post, worker.add_node)
        )
        self.graph_view.sink_changed.connect(
            lambda properties: self.set_view_properties(_BACKDROP, properties)
        )
        self.graph_view.input.message.connect(self.show_message)
        self.properties = QtProperties(self.schedule_properties, _property_pointer)
        self.graph_properties = QtProperties(
            self.schedule_properties, _graph_property_pointer
        )
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
        self.panel = PanelColumn()
        self.log_view = LogView(log_sink)
        self.preferences = preferences
        preferences.shortcut_modifiers_changed.connect(self.set_shortcut_modifiers)
        self.tabs = QtWidgets.QTabWidget()
        for page, name in (
            (self.properties.widget, "Node"),
            (self.graph_properties.widget, "Graph"),
            (self.profiler, "Profiler"),
            (self.memory, "Memory"),
            (self.log_view, "Log"),
            (preferences, "Preferences"),
        ):
            self.tabs.addTab(page, name)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.side = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.side.addWidget(self.graph_view)
        self.side.addWidget(self.tabs)
        self.side.setSizes([350, 550])
        self.split = QtWidgets.QSplitter()
        self.split.addWidget(self.preview)
        self.split.addWidget(self.panel)
        self.split.addWidget(self.side)
        self.split.setSizes([1000, 450, 600])
        self.setCentralWidget(self.split)
        self.veils = [_Veil(self.preview), _Veil(self.graph_view)]
        self.add_menus()
        self.add_footer()
        self.set_shortcut_modifiers(preferences.shortcut_modifiers())
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
        worker.loading.connect(self.set_loading)
        worker.save_captured.connect(self.write_capture)
        QtCore.QTimer(
            self,
            interval=_PROPERTIES_INTERVAL // timedelta(milliseconds=1),
            timeout=self.request_properties,
        ).start()
        QtCore.QTimer(
            self,
            interval=_BUSY_INTERVAL // timedelta(milliseconds=1),
            timeout=self.show_busy,
        ).start()
        self.frame_timer = QtCore.QTimer(
            self,
            interval=_FRAME_INTERVAL // timedelta(milliseconds=1),
            timeout=self.show_frames,
            timerType=QtCore.Qt.TimerType.PreciseTimer,
        )
        self.set_path(path)
        self.set_loading("Loading merian")

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
            "The graph with its backdrop large, the preview above the properties",
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
        view_menu.addSeparator()
        self.release_action = _add_action(
            view_menu,
            "&Release Mouse",
            self.release_mouse,
            tip="Give the mouse back from a graph's view that takes raw input, which a "
            "click into the view captures again",
        )

        playback_menu = self.menuBar().addMenu("&Playback")
        playback_menu.setToolTipsVisible(True)
        self.play_action = _add_action(playback_menu, "&Pause", self.toggle_playing)
        self.play_action.setAutoRepeat(False)
        self.step_action = _add_action(
            playback_menu, "&Step", lambda: self.worker.post(self.worker.step)
        )
        _add_action(
            playback_menu,
            "S&top",
            lambda: self.worker.post(self.worker.stop),
            tip=_STOP_TIP,
        )

    def set_shortcut_modifiers(self, modifiers: QtCore.Qt.KeyboardModifier) -> None:
        release = QtCore.QKeyCombination(modifiers, _RELEASE_MOUSE_KEY)
        self.release_action.setShortcut(QtGui.QKeySequence(release))
        self.view_shortcuts = {release.toCombined()}
        for action, key in zip((self.play_action, self.step_action), _PLAYBACK_KEYS):
            combination = QtCore.QKeyCombination(modifiers, key)
            action.setShortcuts(
                [QtGui.QKeySequence(key), QtGui.QKeySequence(combination)]
            )
            self.view_shortcuts.add(combination.toCombined())
        for view in (self.graph_view, self.preview, *self.panel_views.values()):
            view.input.shortcuts = self.view_shortcuts
        step_tip = f"Step: runs one iteration ({_keys(self.step_action)})"
        self.step_action.setToolTip(step_tip)
        self.step_button.setToolTip(step_tip)
        self.show_playing(self.playing)

    def event(self, event: QtCore.QEvent) -> bool:
        if (
            event.type() == QtCore.QEvent.Type.ShortcutOverride
            and event.key() in _PLAYBACK_KEYS
            and event.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier
            and isinstance(QtWidgets.QApplication.focusWidget(), _PLAYBACK_KEY_USERS)
        ):
            event.accept()
            return True
        return super().event(event)

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

        self.play_button = button("play", "", self.play_action.trigger)
        self.step_button = button("step", "", self.step_action.trigger)
        button("stop", _STOP_TIP, lambda: self.worker.post(self.worker.stop))
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
        self.controls = self.run_controls()
        footer.addWidget(self.controls)
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
        self.busy_label = QtWidgets.QLabel(visible=False)
        self.busy_label.setObjectName("footerCaption")
        footer.addPermanentWidget(self.busy_label)
        footer.addPermanentWidget(stats)

    def set_loading(self, text: str) -> None:
        began, ended = text and not self.loading, self.loading and not text
        self.loading = text
        if began:
            self.held = [
                action
                for menu in self.menuBar().findChildren(QtWidgets.QMenu)
                for action in menu.actions()
                if action.isEnabled()
            ]
        for action in self.held:
            action.setEnabled(not text)
        self.controls.setEnabled(not text)
        for veil in self.veils:
            veil.parentWidget().setEnabled(not text)
            veil.setVisible(bool(text))
            veil.raise_()
        if began:
            self.tabs.setCurrentWidget(self.log_view)
        elif (
            ended
            and self.worker.graph is not None
            and self.tabs.currentWidget() is self.log_view
        ):
            self.tabs.setCurrentWidget(self.profiler)
        self.show_busy()

    def show_busy(self) -> None:
        context = self.worker.context
        text = self.loading or (
            "Compiling shaders" if context and context.compiling_shaders else ""
        )
        self.busy_label.setVisible(bool(text))
        if text:
            self.spin = (self.spin + 1) % len(_SPINNER)
            self.busy_label.setText(f"{_SPINNER[self.spin]} {text}\u2026")

    def show_message(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setToolTip(text)

    def show_playing(self, playing: bool) -> None:
        self.playing = playing
        self.play_button.setIcon(_painted_icon("pause" if playing else "play"))
        self.play_action.setText("&Pause" if playing else "&Play")
        tip = "Pause: holds the graph time" if playing else "Play"
        tip = f"{tip} ({_keys(self.play_action)})"
        self.play_action.setToolTip(tip)
        self.play_button.setToolTip(tip)
        self.step_action.setEnabled(not playing)
        self.step_button.setEnabled(not playing)

    def toggle_playing(self) -> None:
        self.worker.post(self.worker.set_playing, not self.playing)

    def release_mouse(self) -> None:
        for view in self.windows:
            if self.view_of(view).input.release_mouse():
                self.show_message("released the mouse, a click into the view takes it")

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
            self.node_times_resumed = self.node_times_action.isChecked()
            self.node_times_action.setChecked(False)
        elif self.node_times_resumed:
            self.node_times_resumed = False
            self.node_times_action.setChecked(True)

    def refresh(self, state: _GraphState) -> None:
        if state == self.state and not self.new_positions:
            return
        self.state = state
        self.recordings = {
            identifier: items
            for identifier, items in self.recordings.items()
            if identifier is None or identifier in state.nodes
        }
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

    def show_node_time(self, identifier: str) -> None:
        self.tabs.setCurrentWidget(self.profiler)
        self.profiler.show_node(identifier)

    def show_memory(self, identifier: str, output: str) -> None:
        self.graph_view.select_node(identifier)
        self.tabs.setCurrentWidget(self.memory)
        self.memory.show_resource(identifier, output)

    def node_menu(self, identifier: str, position: QtCore.QPoint) -> None:
        nodes = self.state.nodes
        node = nodes[identifier]
        menu = QtWidgets.QMenu(self)
        switch = self.enabled_switch(identifier)
        enabled = switch is None or switch.value
        menu.addAction(
            "Disable" if enabled else "Enable",
            lambda: self.set_node_enabled(identifier, not enabled),
        ).setEnabled(switch is not None)
        menu.addAction(
            "Remove", lambda: self.worker.post(self.worker.delete, [identifier], [])
        )
        menu.addSeparator()
        outputs = menu.addMenu("Outputs")
        for output in node.outputs:
            source = self.output_source(identifier, output.name)
            showable = source.content in _SINK_TYPES
            output_menu = outputs.addMenu(output.name)
            self.add_connect_menu(
                output_menu, identifier, output.name, from_output=True
            )
            output_menu.addAction(
                "Show in Preview",
                functools.partial(self.show_source, _PREVIEW, source),
            ).setEnabled(showable)
            output_menu.addAction(
                "Show in Panel", functools.partial(self.add_panel, source=source)
            ).setEnabled(showable)
        outputs.setEnabled(bool(node.outputs))
        inputs = menu.addMenu("Inputs")
        for input_ in node.inputs:
            self.add_connect_menu(
                inputs.addMenu(input_.name), identifier, input_.name, from_output=False
            )
        inputs.setEnabled(bool(node.inputs))
        menu.exec(position)
        menu.deleteLater()

    def add_connect_menu(
        self, menu: QtWidgets.QMenu, identifier: str, name: str, from_output: bool
    ) -> None:
        nodes = self.state.nodes
        connect = menu.addMenu("Connect To")
        for other in sorted(nodes.keys() - {identifier}):
            connectors = nodes[other].inputs if from_output else nodes[other].outputs
            targets = connect.addMenu(other)
            targets.setEnabled(bool(connectors))
            for connector in connectors:
                ends = (
                    (identifier, name, other, connector.name)
                    if from_output
                    else (other, connector.name, identifier, name)
                )
                targets.addAction(
                    connector.name,
                    functools.partial(
                        self.worker.post, self.worker.add_connection, *ends
                    ),
                )

    def enabled_switch(self, identifier: str):
        return next(
            (
                item
                for item in self.recordings.get(identifier, [])
                if item.kind == "bool" and item.label == "enabled"
            ),
            None,
        )

    def set_node_enabled(self, identifier: str, enabled: bool) -> None:
        key = self.enabled_switch(identifier).key
        self.sequence += 1
        self.change_sequences[identifier] = self.sequence
        self.worker.post(
            self.worker.request_properties,
            self.sequence,
            identifier,
            Recorder({key: enabled}, {}, {}),
        )

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
        menu.addAction(
            "Memory", lambda: self.show_memory(identifier, output)
        ).setToolTip("Where the output's memory is, in the memory view")
        menu.addSeparator()
        views = [view for view in self.shown if self.shows(view, source)]
        buffer = source.content == mg.ConnectorKind.BUFFER
        action = menu.addAction(
            "Save Buffer..." if buffer else "Save Image...",
            lambda: self.save_view(views[0]),
        )
        action.setEnabled(bool(views))
        if not views:
            menu.setToolTipsVisible(True)
            action.setToolTip(
                "Show it in a view first: capturing it reconnects the graph, which "
                "restarts the run"
            )
        menu.exec(position)
        menu.deleteLater()

    def write_capture(self, path: Path, capture: mg.Capture) -> None:
        try:
            save_capture(capture, path)
            self.show_message(f"saved {path}")
        except OSError as error:
            self.show_message(str(error))

    def save_view(self, view: str) -> None:
        if (shown := self.shown.get(view)) is None:
            return
        source = shown.source
        suffix, filters = _SAVE_FILTERS[source.content]
        name = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in source.label()
        )
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save", str(Path.cwd() / f"{name}.{suffix}"), filters
        )
        if not path:
            return
        if source.kind == SourceKind.WINDOW:
            try:
                self.capture_image(view).save(Path(path))
                self.show_message(f"saved {path}")
            except OSError as error:
                self.show_message(str(error))
            return
        properties = self.view_of(view).sink_properties()
        self.worker.post(self.worker.save_capture, view, Path(path), properties)

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

    def capture_image(self, view: str) -> CaptureImage:
        if view == _BACKDROP:
            return self.graph_view.backdrop
        return self.output_view(view).image.capture_image

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
        appearing = self.panel.isHidden()
        self.panel.add(output_view)
        if appearing:
            sizes = self.split.sizes()
            widest = max(range(len(sizes)), key=sizes.__getitem__)
            sizes[self.split.indexOf(self.panel)] = sizes[widest] // 3
            sizes[widest] -= sizes[widest] // 3
            self.split.setSizes(sizes)
        if source is not None:
            self.show_source(view, source)

    def remove_panel(self, view: str) -> None:
        self.hide_source(view)
        output_view = self.panel_views.pop(view)
        self.panel.remove(output_view)
        output_view.deleteLater()

    def connect_view(self, view: str, output_view: OutputView) -> None:
        def close() -> None:
            if view in self.panel_views:
                self.remove_panel(view)
            else:
                self.hide_source(view)

        output_view.closed.connect(close)
        output_view.save_requested.connect(lambda: self.save_view(view))
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
        output_view.input.shortcuts = self.view_shortcuts

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
            if identifier in self.recordings:
                self.properties.show(self.recordings[identifier], consumed=False)
            self.request_properties()

    def schedule_properties(self) -> None:
        if not self.properties_scheduled:
            self.properties_scheduled = True
            QtCore.QTimer.singleShot(0, self.request_properties)

    def request_properties(self) -> None:
        """Records the shown panel, and the other one if it has changes."""
        self.properties_scheduled = False
        if QtWidgets.QApplication.activeModalWidget():
            return
        for identifier in (None, self.selected) if self.selected else (None,):
            panel = self.panel_of(identifier)
            if panel.changes or self.tabs.currentWidget() is panel.widget:
                self.sequence += 1
                recorder = panel.recorder()
                if recorder.changes:
                    self.change_sequences[identifier] = self.sequence
                self.worker.post(
                    self.worker.request_properties, self.sequence, identifier, recorder
                )

    def show_recording(
        self, sequence: int, identifier: str | None, items: list
    ) -> None:
        self.recordings[identifier] = items
        panel = self.panel_of(identifier)
        if sequence == _PUBLISHED:
            if panel is not None and panel.empty:
                panel.show(items, consumed=False)
        elif panel is not None:
            panel.show(items, sequence >= self.change_sequences.get(identifier, 0))

    def panel_of(self, identifier: str | None) -> QtProperties | None:
        if identifier is None:
            return self.graph_properties
        return self.properties if identifier == self.selected else None

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
        self.recordings = {}
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
    preferences = PreferencesView(QtCore.QSettings(*_SETTINGS))
    with merian.LogSink() as log_sink:
        worker = _GraphWorker(
            lambda: scene.context((mg.HOST_WINDOWS_EXTENSION,)), description
        )
        window = _Window(worker, path, new_positions, log_sink, preferences)
        window.showMaximized()
        worker.start()
        try:
            return application.exec()
        finally:
            window.hide()
            worker.quit()


if __name__ == "__main__":
    sys.exit(main())
