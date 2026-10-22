import bisect
import itertools
import math
import time
import zlib
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from PySide6 import QtCore, QtGui, QtWidgets

import merian
import merian.graph as mg

from .theme import RIGHT_ALIGNED, THEME, scrolled

_BLOCK_HEIGHT = 200
_MIN_SPAN_HEIGHT = 7
_HISTORY = 300
_RUN_NODES = "Run nodes"
_REPORT_KINDS = {
    "interval": mg.ReportKind.INTERVAL,
    "run": mg.ReportKind.RUN,
    "build": mg.ReportKind.BUILD,
}
_SECTION_COLORS = ("accent", "peach", "blue", "green", "yellow", "red", "teal")
_TOTAL = ()
_UNIT_HOLD = 1.0

_NODE_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2
_PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole
_SHARE_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_STOPPED_ROLE = QtCore.Qt.ItemDataRole.UserRole + 3

WIDEST_SIZE = "8888.8 MiB"


def format_size(size: float) -> str:
    for unit in ("B", "KiB", "MiB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.2f} GiB"


def format_ms(milliseconds: float) -> str:
    return (
        f"{milliseconds * 1000:.3g} \u00b5s"
        if 0 < milliseconds < 0.1
        else f"{milliseconds:.3g} ms"
    )


class TimeFormat:
    def __init__(self):
        self.micro = False
        self.other_since: float | None = None

    def __call__(self, milliseconds: float) -> str:
        micro = milliseconds < 0.1
        if micro == self.micro:
            self.other_since = None
        elif self.other_since is None:
            self.other_since = time.monotonic()
        elif time.monotonic() - self.other_since >= _UNIT_HOLD:
            self.micro, self.other_since = micro, None
        if self.micro:
            return f"{milliseconds * 1000:.2f} µs"
        return f"{milliseconds:.2f} ms"


def _nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(value))
    return next(step * power for step in (1, 2, 2.5, 5, 10) if step * power >= value)


def _name_color(name: str) -> QtGui.QColor:
    return QtGui.QColor.fromHsv(zlib.crc32(name.encode()) % 360, 110, 230)


def _count(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


class _Series(NamedTuple):
    name: str
    color: str
    # Oldest first, None where there is no sample
    values: list[float | None]
    dashed: bool = False


class _HistoryPlot(QtWidgets.QWidget):
    def __init__(self, format_value: Callable[[float], str]):
        super().__init__()
        self.format_value = format_value
        self.capacity = _HISTORY
        self.series: list[_Series] = []
        self.hovered: int | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(110)

    def set_series(self, series: list[_Series], capacity: int | None = None) -> None:
        self.capacity = capacity or self.capacity
        self.series = series
        self.update()

    def top(self) -> float:
        return _nice_ceiling(
            max(
                (
                    value
                    for entry in self.series
                    for value in entry.values
                    if value is not None
                ),
                default=0,
            )
        )

    def labels(self) -> list[str]:
        return [self.format_value(self.top() * step / 4) for step in range(5)]

    def legend(self) -> list[tuple[QtCore.QPointF, str, _Series]]:
        metrics = self.fontMetrics()
        age = self.hovered or 0
        entries, x, y = [], 4.0, 4.0
        for entry in self.series:
            value = entry.values[-1 - age] if age < len(entry.values) else None
            text = (
                f"{entry.name} {self.format_value(value) if value is not None else '-'}"
            )
            width = 28 + metrics.horizontalAdvance(text)
            if x > 4 and x + width > self.width():
                x, y = 4.0, y + metrics.height() + 2
            entries.append((QtCore.QPointF(x, y), text, entry))
            x += width
        return entries

    def plot_rect(self) -> QtCore.QRectF:
        metrics = self.fontMetrics()
        left = max(metrics.horizontalAdvance(label) for label in self.labels())
        legend = self.legend()
        top = (legend[-1][0].y() if legend else 4) + metrics.height() + 4
        return QtCore.QRectF(self.rect()).adjusted(left + 10, top, -8, -6)

    def x_of(self, rect: QtCore.QRectF, age: int) -> float:
        return rect.right() - age * rect.width() / max(self.capacity - 1, 1)

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        metrics = self.fontMetrics()
        rect = self.plot_rect()
        top = self.top()
        for step, label in enumerate(self.labels()):
            y = rect.bottom() - step / 4 * rect.height()
            painter.setPen(QtGui.QColor(THEME["surface_hi"]))
            painter.drawLine(
                QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y)
            )
            painter.setPen(QtGui.QColor(THEME["text_dim"]))
            label_rect = QtCore.QRectF(
                0, y - metrics.height() / 2, rect.left() - 6, metrics.height()
            )
            painter.drawText(label_rect, RIGHT_ALIGNED, label)
        for entry in self.series:
            path = QtGui.QPainterPath()
            drawing = False
            for age, value in enumerate(reversed(entry.values)):
                if value is None:
                    drawing = False
                    continue
                point = QtCore.QPointF(
                    self.x_of(rect, age), rect.bottom() - value / top * rect.height()
                )
                if drawing:
                    path.lineTo(point)
                else:
                    path.moveTo(point)
                drawing = True
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(entry.color),
                    1.5,
                    QtCore.Qt.PenStyle.DashLine
                    if entry.dashed
                    else QtCore.Qt.PenStyle.SolidLine,
                )
            )
            painter.drawPath(path)
        if self.hovered is not None:
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(THEME["text_dim"]), 1, QtCore.Qt.PenStyle.DashLine
                )
            )
            x = self.x_of(rect, self.hovered)
            painter.drawLine(
                QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom())
            )
        for position, text, entry in self.legend():
            y = position.y() + metrics.height() / 2
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(entry.color),
                    2,
                    QtCore.Qt.PenStyle.DotLine
                    if entry.dashed
                    else QtCore.Qt.PenStyle.SolidLine,
                )
            )
            painter.drawLine(
                QtCore.QPointF(position.x(), y), QtCore.QPointF(position.x() + 10, y)
            )
            painter.setPen(QtGui.QColor(THEME["text"]))
            painter.drawText(
                QtCore.QPointF(position.x() + 14, position.y() + metrics.ascent()), text
            )

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        rect = self.plot_rect()
        age = round(
            (rect.right() - event.position().x()) / rect.width() * (self.capacity - 1)
        )
        longest = max((len(entry.values) for entry in self.series), default=0)
        self.hovered = age if 0 <= age < longest else None
        self.update()

    def leaveEvent(self, _: QtCore.QEvent) -> None:
        self.hovered = None
        self.update()


class _BarDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, color: str, parent: QtCore.QObject):
        super().__init__(parent)
        self.color = QtGui.QColor(color)
        self.color.setAlpha(70)

    def paint(self, painter: QtGui.QPainter, option, index: QtCore.QModelIndex) -> None:
        if share := index.data(_SHARE_ROLE):
            bar = QtCore.QRectF(option.rect).adjusted(1, 3, -1, -3)
            bar.setWidth(bar.width() * min(share, 1.0))
            painter.fillRect(bar, self.color)
        super().paint(painter, option, index)


class _StatCard(QtWidgets.QFrame):
    def __init__(self, title: str, color: str = "text"):
        super().__init__()
        self.setObjectName("card")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(0)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("cardTitle")
        self.value = QtWidgets.QLabel("-")
        self.value.setObjectName("cardValue")
        self.value.setStyleSheet(f"color: {THEME[color]};")
        self.note = QtWidgets.QLabel()
        self.note.setObjectName("cardNote")
        for label in (heading, self.value, self.note):
            layout.addWidget(label)

    def set(self, value: str, note: str = "", tooltip: str = "") -> None:
        self.value.setText(value)
        self.note.setText(note)
        self.setToolTip(tooltip)


def _card_row(*cards: _StatCard) -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(8)
    for card in cards:
        row.addWidget(card, 1)
    return row


# Profiler
##########


@dataclass
class _Section:
    """A profiled section: CPU and GPU (mean, standard deviation) in ms."""

    cpu: tuple[float, float] | None = None
    gpu: tuple[float, float] | None = None
    stopped: bool = True
    children: dict[str, "_Section"] = field(default_factory=dict)


def _merged_sections(report: merian.ProfilerReport) -> dict[str, _Section]:
    sections: dict[str, _Section] = {}

    def add(entries: list, side: str, into: dict[str, _Section]) -> None:
        for entry in entries:
            section = into.setdefault(entry.name, _Section())
            setattr(section, side, (entry.duration_ms, entry.std_deviation_ms))
            section.stopped &= entry.last_seen_ms_ago is not None
            add(entry.children, side, section.children)

    add(report.cpu, "cpu", sections)
    add(report.gpu, "gpu", sections)
    return sections


def _section_times(
    sections: dict[str, _Section], prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], tuple]:
    times = {}
    for name, section in sections.items():
        path = (*prefix, name)
        times[path] = (section.cpu and section.cpu[0], section.gpu and section.gpu[0])
        times |= _section_times(section.children, path)
    return times


def node_times(report: merian.ProfilerReport) -> dict[str, float]:
    """Returns the GPU time in ms of each node that ran, by identifier."""
    run = next((entry for entry in report.gpu if entry.name == _RUN_NODES), None)
    if run is None:
        return {}
    return {
        entry.name.partition(" (")[0]: entry.duration_ms
        for entry in run.children
        if entry.last_seen_ms_ago is None
    }


class ProfilerView(QtWidgets.QWidget):
    enabled_changed = QtCore.Signal(bool)
    kind_changed = QtCore.Signal(object)
    # In ms
    interval_changed = QtCore.Signal(int)
    restart_requested = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.enabled = QtWidgets.QCheckBox("Profile")
        self.enabled.toggled.connect(self.enabled_changed)
        self.kind = QtWidgets.QComboBox()
        self.kind.addItems(list(_REPORT_KINDS))
        self.kind.setToolTip(
            "interval: the last report interval, run: since the connect or the "
            "restart, build: the last connect"
        )
        self.kind.currentTextChanged.connect(
            lambda text: self.kind_changed.emit(_REPORT_KINDS[text])
        )
        self.interval = QtWidgets.QSpinBox(
            minimum=10,
            maximum=10000,
            singleStep=10,
            value=50,
            suffix=" ms",
            keyboardTracking=False,
        )
        self.interval.setToolTip(
            "Report interval: the period of the means and deviations"
        )
        self.interval.valueChanged.connect(self.interval_changed)
        self.cpu = _StatCard("CPU", "accent")
        self.gpu = _StatCard("GPU", "peach")
        self.rate = _StatCard("Frame rate")
        self.selected_card = _StatCard("Selected node")
        self.selected: str | None = None
        self.node_times: dict[str, float] = {}
        self.gpu_total = 0.0
        self.formats = {
            name: TimeFormat()
            for name in ("cpu", "cpu std", "gpu", "gpu std", "frame", "node")
        }
        restart = QtWidgets.QToolButton(
            text="Restart", toolTip="Restarts the run report"
        )
        restart.clicked.connect(self.restart_requested)
        controls = QtWidgets.QHBoxLayout()
        for widget in (self.enabled, self.kind, self.interval, restart):
            controls.addWidget(widget)
        controls.addStretch()

        self.plot = _HistoryPlot(format_ms)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["section", "CPU ms", "\u00b1", "GPU ms", "\u00b1"])
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(True)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column in range(1, 5):
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        for column, color in ((1, THEME["accent"]), (3, THEME["peach"])):
            self.tree.setItemDelegateForColumn(column, _BarDelegate(color, self.tree))
        self.tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.tree.setToolTip(
            "Select sections to plot their history, several with Ctrl or Shift"
        )
        self.tree.itemSelectionChanged.connect(self.plot_selected)
        self.items: dict[tuple[str, ...], QtWidgets.QTreeWidgetItem] = {}
        self.history: deque[dict] = deque(maxlen=_HISTORY)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(self.plot)
        split.addWidget(self.tree)
        split.setSizes([160, 400])
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(_card_row(self.cpu, self.gpu, self.rate, self.selected_card))
        layout.addWidget(split, 1)
        self.plot_selected()

    def set_selected(self, node: str | None) -> None:
        if node != self.selected:
            self.formats["node"] = TimeFormat()
        self.selected = node
        self.show_selected_time()
        for path, item in self.items.items():
            self.mark(path, item)

    def show_selected_time(self) -> None:
        if self.selected is None:
            self.selected_card.set("-", "select a node")
        elif (milliseconds := self.node_times.get(self.selected)) is None:
            self.selected_card.set(self.selected, "did not run")
        else:
            share = milliseconds / self.gpu_total if self.gpu_total else 0.0
            self.selected_card.set(
                self.selected, f"{self.formats['node'](milliseconds)} GPU, {share:.0%}"
            )

    def mark(self, path: tuple[str, ...], item: QtWidgets.QTreeWidgetItem) -> None:
        marked = self.selected is not None and path[-1].startswith(f"{self.selected} (")
        stopped = item.data(0, _STOPPED_ROLE)
        font = item.font(0)
        newly_marked = marked and not font.bold()
        font.setBold(marked)
        color = QtGui.QColor(THEME["text_dim" if stopped else "text"])
        for column in range(5):
            item.setFont(column, font)
            item.setForeground(column, color)
        if marked and not stopped:
            item.setForeground(0, QtGui.QColor(THEME["accent_hi"]))
        if newly_marked:
            self.tree.scrollToItem(item)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled.setChecked(enabled)

    def set_interval(self, milliseconds: int) -> None:
        if not self.interval.hasFocus():
            with QtCore.QSignalBlocker(self.interval):
                self.interval.setValue(milliseconds)

    def selected_paths(self) -> list[tuple[str, ...]]:
        paths = [item.data(0, _PATH_ROLE) for item in self.tree.selectedItems()]
        return paths[: len(_SECTION_COLORS)] or [_TOTAL]

    def add_sample(self, report: merian.ProfilerReport) -> None:
        times = _section_times(_merged_sections(report))
        times[_TOTAL] = (report.cpu_total_ms(), report.gpu_total_ms())
        self.history.append(times)
        self.plot_selected()

    def plot_selected(self) -> None:
        paths = self.selected_paths()
        series = []
        for index, path in enumerate(paths):
            samples = [times.get(path, (None, None)) for times in self.history]
            name = path[-1] if path else "total"
            cpu_color = THEME[_SECTION_COLORS[index]]
            gpu_color = THEME[_SECTION_COLORS[1]] if len(paths) == 1 else cpu_color
            series += [
                _Series(f"{name} CPU", cpu_color, [cpu for cpu, _ in samples]),
                _Series(
                    f"{name} GPU",
                    gpu_color,
                    [gpu for _, gpu in samples],
                    dashed=len(paths) > 1,
                ),
            ]
        self.plot.set_series(series)

    def show_report(self, report: merian.ProfilerReport) -> None:
        totals = (report.cpu_total_ms(), report.gpu_total_ms())
        formats = self.formats
        self.cpu.set(
            formats["cpu"](totals[0]),
            f"\u00b1 {formats['cpu std'](report.cpu_total_std_deviation_ms())}",
        )
        self.gpu.set(
            formats["gpu"](totals[1]),
            f"\u00b1 {formats['gpu std'](report.gpu_total_std_deviation_ms())}",
        )
        frame = max(totals)
        self.rate.set(
            f"{1000 / frame:.0f} fps" if frame > 0 else "-",
            f"{formats['frame'](frame)} per frame",
            "Bounded by the slower of CPU and GPU",
        )
        self.node_times, self.gpu_total = node_times(report), totals[1]
        self.show_selected_time()
        seen = set()

        def update(
            parent: QtWidgets.QTreeWidgetItem,
            sections: dict[str, _Section],
            prefix: tuple[str, ...],
        ) -> None:
            for name, section in sections.items():
                path = (*prefix, name)
                seen.add(path)
                if (item := self.items.get(path)) is None:
                    item = QtWidgets.QTreeWidgetItem([name])
                    item.setData(0, _PATH_ROLE, path)
                    for column in range(1, 5):
                        item.setTextAlignment(column, RIGHT_ALIGNED)
                    parent.addChild(item)
                    item.setExpanded(not prefix)
                    self.items[path] = item
                if item.data(0, _STOPPED_ROLE) != section.stopped:
                    item.setData(0, _STOPPED_ROLE, section.stopped)
                    self.mark(path, item)
                for column, times, total in (
                    (1, section.cpu, totals[0]),
                    (3, section.gpu, totals[1]),
                ):
                    mean, deviation = times or (None, None)
                    item.setText(column, f"{mean:.3f}" if mean is not None else "")
                    item.setText(
                        column + 1, f"{deviation:.3f}" if deviation is not None else ""
                    )
                    item.setData(
                        column, _SHARE_ROLE, mean / total if mean and total else 0.0
                    )
                update(item, section.children, path)

        update(self.tree.invisibleRootItem(), _merged_sections(report), ())
        for path in sorted(self.items.keys() - seen, key=len, reverse=True):
            item = self.items.pop(path)
            (item.parent() or self.tree.invisibleRootItem()).removeChild(item)


# Memory
########


# The colors of the kinds of memory in the order they stack
_KIND_COLORS = {
    "transient": "blue",
    "persistent": "teal",
    "exclusive": "yellow",
    "own": "peach",
}
_KIND_NOTES = {
    "exclusive": "outputs with an allocation each, outside the blocks",
    "own": "allocated by the node itself",
    "persistent": "outputs in the blocks that are kept over the period",
    "transient": "outputs in the blocks, may share memory with each other",
}


def _node_kinds(report: dict) -> dict[str, dict[str, int]]:
    kinds = {
        allocation["node"]: dict.fromkeys(_KIND_COLORS, 0)
        for allocation in report["node_allocations"]
    }

    def of(node: str) -> dict[str, int]:
        return kinds.setdefault(node, dict.fromkeys(_KIND_COLORS, 0))

    for entry in report["transient"]:
        of(entry["node"])["persistent" if entry["persistent"] else "transient"] += (
            entry["size"]
        )
    for entry in report["exclusive"]:
        of(entry["node"])["exclusive"] += entry["size"]
    for allocation in report["node_allocations"]:
        node = kinds[allocation["node"]]
        node["own"] = max(allocation["size"] - node["exclusive"], 0)
    return kinds


class _NodeBars(QtWidgets.QWidget):
    clicked = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.rows: list[tuple[str, dict[str, int]]] = []
        self.selected: str | None = None
        self.hovered: int | None = None
        self.setMouseTracking(True)

    def row_height(self) -> int:
        return self.fontMetrics().height() + 12

    def set_rows(self, rows: list[tuple[str, dict[str, int]]]) -> None:
        self.rows = rows
        self.setMinimumHeight((len(rows) + 1) * self.row_height() + 8)
        self.update()

    def set_selected(self, node: str | None) -> None:
        self.selected = node
        self.update()

    def row_at(self, y: float) -> int | None:
        row = int(y // self.row_height()) - 1
        return row if 0 <= row < len(self.rows) else None

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        metrics = self.fontMetrics()
        height = self.row_height()
        x = 6.0
        for kind, color in _KIND_COLORS.items():
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(THEME[color]))
            painter.drawRoundedRect(QtCore.QRectF(x, height / 2 - 5, 10, 10), 3, 3)
            painter.setPen(QtGui.QColor(THEME["text_dim"]))
            painter.drawText(
                QtCore.QPointF(x + 15, (height + metrics.ascent()) / 2 - 2), kind
            )
            x += 32 + metrics.horizontalAdvance(kind)
        if not self.rows:
            return
        label_width = min(
            max(metrics.horizontalAdvance(node) for node, _ in self.rows) + 20,
            self.width() * 0.35,
        )
        sum_width = metrics.horizontalAdvance(WIDEST_SIZE) + 16
        bar_width = max(self.width() - label_width - sum_width - 8, 1)
        largest = max(sum(kinds.values()) for _, kinds in self.rows) or 1
        for row, (node, kinds) in enumerate(self.rows):
            y = (row + 1) * height
            if node == self.selected or row == self.hovered:
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor(THEME["raised"]))
                painter.drawRoundedRect(
                    QtCore.QRectF(0, y + 1, self.width(), height - 2), 6, 6
                )
            selected = node == self.selected
            painter.setPen(QtGui.QColor(THEME["accent_hi" if selected else "text"]))
            painter.drawText(
                QtCore.QRectF(8, y, label_width - 12, height),
                QtCore.Qt.AlignmentFlag.AlignVCenter,
                metrics.elidedText(
                    node, QtCore.Qt.TextElideMode.ElideMiddle, round(label_width - 12)
                ),
            )
            total = sum(kinds.values())
            bar = QtCore.QRectF(
                label_width, y + 6, max(total / largest * bar_width, 1), height - 12
            )
            clip = QtGui.QPainterPath()
            clip.addRoundedRect(bar, 3, 3)
            painter.save()
            painter.setClipPath(clip)
            x = bar.left()
            for kind, color in _KIND_COLORS.items():
                width = kinds[kind] / largest * bar_width
                painter.fillRect(
                    QtCore.QRectF(x, bar.top(), width, bar.height()),
                    QtGui.QColor(THEME[color]),
                )
                x += width
            painter.restore()
            painter.setPen(QtGui.QColor(THEME["text_dim"]))
            painter.drawText(
                QtCore.QRectF(bar.right() + 8, y, sum_width, height),
                QtCore.Qt.AlignmentFlag.AlignVCenter,
                format_size(total),
            )

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        row = self.row_at(event.position().y())
        if row != self.hovered:
            self.hovered = row
            self.update()
        if row is None:
            QtWidgets.QToolTip.hideText()
            return
        node, kinds = self.rows[row]
        lines = [node] + [
            f"{kind}: {format_size(size)}" for kind, size in kinds.items() if size
        ]
        QtWidgets.QToolTip.showText(
            event.globalPosition().toPoint(), "\n".join(lines), self
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (row := self.row_at(event.position().y())) is not None:
            self.clicked.emit(self.rows[row][0])

    def leaveEvent(self, _: QtCore.QEvent) -> None:
        self.hovered = None
        self.update()


def _runs(positions: list[int]) -> list[tuple[int, int]]:
    spans = []
    for position in sorted(positions):
        if spans and spans[-1][1] == position:
            spans[-1] = (spans[-1][0], position + 1)
        else:
            spans.append((position, position + 1))
    return spans


class _BlockMap(QtWidgets.QWidget):
    def __init__(
        self, block: dict, transient: list[dict], layer_count: int, period: int
    ):
        super().__init__()
        self.block = block
        self.transient = transient
        self.layer_count = max(layer_count, 1)
        self.positions = max(layer_count * period, 1)
        self.selected: str | None = None
        self.hovered: dict | None = None
        self.setMouseTracking(True)
        size = max(block["size"], 1)
        self.boundaries = sorted(
            {0, size}
            | {entry["offset"] for entry in transient}
            | {entry["offset"] + entry["size"] for entry in transient}
        )
        self.heights_below = list(
            itertools.accumulate(
                (
                    max((upper - lower) / size * _BLOCK_HEIGHT, _MIN_SPAN_HEIGHT)
                    for lower, upper in itertools.pairwise(self.boundaries)
                ),
                initial=0.0,
            )
        )
        metrics = self.fontMetrics()
        self.setFixedHeight(round(self.heights_below[-1] + 1.5 * metrics.height() + 7))

    def set_selected(self, node: str | None) -> None:
        self.selected = node
        self.update()

    def map_rect(self) -> QtCore.QRectF:
        metrics = self.fontMetrics()
        left = metrics.horizontalAdvance(WIDEST_SIZE) + 10
        return QtCore.QRectF(self.rect()).adjusted(
            left, metrics.height() / 2 + 1, -1, -metrics.height() - 6
        )

    def y_of(self, rect: QtCore.QRectF, offset: float) -> float:
        """The y of an offset into the block, linear between resource boundaries."""
        index = min(
            max(bisect.bisect_right(self.boundaries, offset) - 1, 0),
            len(self.boundaries) - 2,
        )
        lower, upper = self.boundaries[index], self.boundaries[index + 1]
        fraction = (offset - lower) / (upper - lower)
        height = self.heights_below[index] + fraction * (
            self.heights_below[index + 1] - self.heights_below[index]
        )
        return rect.bottom() - height / self.heights_below[-1] * rect.height()

    def entry_rects(self, entry: dict) -> list[QtCore.QRectF]:
        rect = self.map_rect()
        top = self.y_of(rect, entry["offset"] + entry["size"])
        height = max(self.y_of(rect, entry["offset"]) - top, 1)
        positions = range(self.positions) if entry["persistent"] else entry["in_use"]
        return [
            QtCore.QRectF(
                rect.left() + first / self.positions * rect.width(),
                top,
                (last - first) / self.positions * rect.width(),
                height,
            )
            for first, last in _runs(list(positions))
        ]

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        rect = self.map_rect()
        metrics = self.fontMetrics()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(THEME["surface"]))
        painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(QtGui.QColor(THEME["text_dim"]))
        for fraction in (0.0, 0.5, 1.0):
            y = self.y_of(rect, fraction * self.block["size"])
            painter.drawText(
                QtCore.QRectF(
                    0, y - metrics.height() / 2, rect.left() - 8, metrics.height()
                ),
                RIGHT_ALIGNED,
                format_size(fraction * self.block["size"]),
            )
        # Every step-th layer of each iteration
        step = max(1, math.ceil(self.positions * 28 / max(rect.width(), 1)))
        for position in range(self.positions):
            if position % self.layer_count % step:
                continue
            x = rect.left() + (position + 0.5) / self.positions * rect.width()
            painter.drawText(
                QtCore.QRectF(x - 20, rect.bottom() + 3, 40, metrics.height()),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                str(position % self.layer_count),
            )
        dashed = QtGui.QPen(
            QtGui.QColor(THEME["surface_hi"]), 1, QtCore.Qt.PenStyle.DashLine
        )
        painter.setPen(dashed)
        for iteration in range(1, self.positions // self.layer_count):
            x = (
                rect.left()
                + iteration * self.layer_count / self.positions * rect.width()
            )
            painter.drawLine(
                QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom())
            )

        for entry in self.transient:
            color = _name_color(entry["node"])
            fill = QtGui.QColor(color)
            emphasized = self.selected is None or entry["node"] == self.selected
            fill.setAlpha(190 if emphasized else 45)
            outline = (
                QtGui.QColor(THEME["accent_hi"])
                if entry is self.hovered
                else color.darker(170)
            )
            painter.setPen(QtGui.QPen(outline, 2 if entry is self.hovered else 1))
            painter.setBrush(fill)
            for entry_rect in self.entry_rects(entry):
                painter.drawRect(entry_rect)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        position = event.position()
        hovered = next(
            (
                entry
                for entry in reversed(self.transient)
                if any(rect.contains(position) for rect in self.entry_rects(entry))
            ),
            None,
        )
        if hovered is not self.hovered:
            self.hovered = hovered
            self.update()
        if hovered is None:
            QtWidgets.QToolTip.hideText()
            return
        flags = [name for name in ("aliased", "persistent", "scratch") if hovered[name]]
        lines = [
            f"{hovered['node']}.{hovered['output']}",
            f"{format_size(hovered['size'])} at {format_size(hovered['offset'])}",
        ]
        QtWidgets.QToolTip.showText(
            event.globalPosition().toPoint(),
            "\n".join(lines + ([", ".join(flags)] if flags else [])),
            self,
        )

    def leaveEvent(self, _: QtCore.QEvent) -> None:
        self.hovered = None
        self.update()


class MemoryView(QtWidgets.QWidget):
    node_clicked = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.report: dict | None = None
        self.selected: str | None = None
        self.total = _StatCard("Total")
        self.cards = {
            kind: _StatCard(kind.capitalize(), _KIND_COLORS[kind])
            for kind in ("transient", "exclusive", "own")
        }

        self.nodes = _NodeBars()
        self.nodes.clicked.connect(self.node_clicked)

        self.in_use = _HistoryPlot(format_size)
        self.block_maps: list[_BlockMap] = []
        self.blocks = QtWidgets.QVBoxLayout()
        blocks = QtWidgets.QWidget()
        blocks_layout = QtWidgets.QVBoxLayout(blocks)
        blocks_layout.addLayout(self.blocks)
        in_use_title = QtWidgets.QLabel("In use at each layer of the period")
        in_use_title.setObjectName("cardTitle")
        blocks_layout.addWidget(in_use_title)
        blocks_layout.addWidget(self.in_use)
        blocks_layout.addStretch()

        self.search = QtWidgets.QLineEdit(placeholderText="Filter resources")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter_resources)
        self.resources = QtWidgets.QTreeWidget()
        self.resources.setHeaderLabels(["resource", "size", "kind", "where"])
        self.resources.setUniformRowHeights(True)
        self.resources.setItemDelegateForColumn(
            1, _BarDelegate(THEME["blue"], self.resources)
        )
        header = self.resources.header()
        for column in range(3):
            header.setSectionResizeMode(
                column, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
            )
        self.resources.itemClicked.connect(
            lambda item: self.node_clicked.emit(item.data(0, _NODE_ROLE))
        )
        resources = QtWidgets.QWidget()
        resources_layout = QtWidgets.QVBoxLayout(resources)
        resources_layout.setContentsMargins(0, 6, 0, 0)
        resources_layout.addWidget(self.search)
        resources_layout.addWidget(self.resources, 1)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(scrolled(self.nodes), "Nodes")
        tabs.addTab(scrolled(blocks), "Blocks")
        tabs.addTab(resources, "Resources")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(_card_row(self.total, *self.cards.values()))
        layout.addWidget(tabs, 1)

    def set_selected(self, node: str | None) -> None:
        self.selected = node
        self.nodes.set_selected(node)
        for block_map in self.block_maps:
            block_map.set_selected(node)
        for row in range(self.resources.topLevelItemCount()):
            item = self.resources.topLevelItem(row)
            if item.text(0) == node:
                item.setExpanded(True)
                self.resources.setCurrentItem(item)
                self.resources.scrollToItem(item)

    def show_report(self, report: dict) -> None:
        if report == self.report:
            return
        self.report = report
        kinds = _node_kinds(report)
        totals = {
            kind: sum(node[kind] for node in kinds.values()) for kind in _KIND_COLORS
        }
        block_size = sum(block["size"] for block in report["blocks"])
        in_blocks = totals["transient"] + totals["persistent"]
        saved = max(in_blocks - block_size, 0)
        self.total.set(
            format_size(report["total_size"]),
            f"{_count(len(kinds), 'node')}, aliasing {'on' if report['aliasing'] else 'off'}",
        )
        self.cards["transient"].set(
            format_size(block_size),
            f"{_count(len(report['blocks']), 'block')}, aliasing saves {format_size(saved)}",
            f"Blocks of transient memory: {_KIND_NOTES['transient']}, and "
            f"{format_size(totals['persistent'])} {_KIND_NOTES['persistent']}",
        )
        self.cards["exclusive"].set(
            format_size(totals["exclusive"]),
            _count(len(report["exclusive"]), "output"),
            _KIND_NOTES["exclusive"].capitalize(),
        )
        self.cards["own"].set(
            format_size(totals["own"]),
            _count(sum(1 for node in kinds.values() if node["own"]), "node"),
            _KIND_NOTES["own"].capitalize(),
        )
        self.nodes.set_rows(
            sorted(kinds.items(), key=lambda entry: -sum(entry[1].values()))
        )
        self.show_blocks(report)
        self.show_resources(report, kinds)
        self.set_selected(self.selected)

    def show_blocks(self, report: dict) -> None:
        in_use = report["in_use"]
        self.in_use.set_series(
            [_Series("in use", THEME["blue"], list(in_use))],
            capacity=max(len(in_use), 2),
        )
        while self.blocks.count():
            self.blocks.takeAt(0).widget().deleteLater()
        self.block_maps = []
        for index, block in enumerate(report["blocks"]):
            title = QtWidgets.QLabel(
                f"<b>Block {index}</b>&nbsp;&nbsp;<span style='color: {THEME['text_dim']}'>"
                f"{format_size(block['size'])} holding "
                f"{format_size(block['requested_size'])} of resources, memory type "
                f"{block['memory_type_index']}</span>"
            )
            self.blocks.addWidget(title)
            transient = [
                entry for entry in report["transient"] if entry["block"] == index
            ]
            nodes = dict.fromkeys(entry["node"] for entry in transient)
            legend = QtWidgets.QLabel(
                "&nbsp;&nbsp;&nbsp;".join(
                    f"<span style='color: {_name_color(node).name()}'>\u25a0</span> {node}"
                    for node in nodes
                )
            )
            legend.setWordWrap(True)
            self.blocks.addWidget(legend)
            block_map = _BlockMap(
                block, transient, report["layer_count"], report["period"]
            )
            self.block_maps.append(block_map)
            self.blocks.addWidget(block_map)

    def show_resources(self, report: dict, kinds: dict[str, dict[str, int]]) -> None:
        children: dict[str, list[tuple[str, int, str, str]]] = defaultdict(list)
        for entry in report["transient"]:
            flags = [name for name in ("aliased", "scratch") if entry[name]]
            where = f"block {entry['block']} at {format_size(entry['offset'])}"
            children[entry["node"]].append(
                (
                    entry["output"],
                    entry["size"],
                    "persistent" if entry["persistent"] else "transient",
                    ", ".join([where, *flags]),
                )
            )
        for entry in report["exclusive"]:
            children[entry["node"]].append(
                (entry["output"], entry["size"], "exclusive", entry["reason"])
            )
        for node, node_kinds in kinds.items():
            if node_kinds["own"]:
                children[node].append(("internal", node_kinds["own"], "own", ""))
        totals = {
            node: sum(child[1] for child in entries)
            for node, entries in children.items()
        }
        largest = max(totals.values(), default=0) or 1
        expanded = {
            self.resources.topLevelItem(row).text(0)
            for row in range(self.resources.topLevelItemCount())
            if self.resources.topLevelItem(row).isExpanded()
        }
        self.resources.clear()
        for node in sorted(totals, key=lambda node: -totals[node]):
            item = QtWidgets.QTreeWidgetItem([node, format_size(totals[node])])
            item.setData(0, _NODE_ROLE, node)
            item.setData(1, _SHARE_ROLE, totals[node] / largest)
            for name, size, kind, where in sorted(
                children[node], key=lambda child: -child[1]
            ):
                child = QtWidgets.QTreeWidgetItem(
                    [name, format_size(size), kind, where]
                )
                child.setData(0, _NODE_ROLE, node)
                child.setData(1, _SHARE_ROLE, size / largest)
                child.setForeground(2, QtGui.QColor(THEME[_KIND_COLORS[kind]]))
                child.setToolTip(2, _KIND_NOTES[kind])
                item.addChild(child)
            self.resources.addTopLevelItem(item)
            item.setExpanded(node in expanded)
        self.filter_resources(self.search.text())

    def filter_resources(self, text: str) -> None:
        text = text.strip().lower()
        for row in range(self.resources.topLevelItemCount()):
            item = self.resources.topLevelItem(row)
            node_matches = text in item.text(0).lower()
            shown = 0
            for index in range(item.childCount()):
                child = item.child(index)
                visible = node_matches or any(
                    text in child.text(column).lower() for column in (0, 2, 3)
                )
                child.setHidden(not visible)
                shown += visible
            item.setHidden(bool(text) and not shown)
            if text and shown:
                item.setExpanded(True)
