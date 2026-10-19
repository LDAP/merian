import dataclasses
import functools
import json
import math
from collections.abc import Callable
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

import merian
import merian.graph as mg

from .theme import NODE_STATE_COLORS, THEME, scrolled

_DEFAULT_DECIMALS = 3
_INDENT = 16
_MAX_SEGMENTS = 4
_MAX_SEGMENT_LENGTH = 14
_MAX_SLIDER_SPAN = 10000
_MAX_SLIDER_STEPS = 1000
_NOTE_COLORS = {"Errors:": "red", "Force disabled:": "peach", "Unsupported:": "yellow"}
_SPACING = 2
_TITLE_STATES = {
    "ERROR": mg.NodeState.ERRORED,
    "F-DISABLED": mg.NodeState.FORCE_DISABLED,
}


# Recording
###########


@dataclasses.dataclass(slots=True)
class _Item:
    """A recorded property; an opened child is followed by its items and "end"."""

    kind: str
    key: tuple
    label: str = ""
    value: object = None
    desc: str = ""
    details: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class _Scope:
    path: tuple
    holds_tabs: bool
    occurrences: dict = dataclasses.field(default_factory=dict)
    first_tab: tuple | None = None


class Recorder(merian.Properties):
    """
    Records one pass and returns the changes it was given. expanded {path: bool}
    overrides DEFAULT_OPEN, tabs {TABS child path: current tab path} the first tab.
    """

    def __init__(
        self, changes: dict, expanded: dict, tabs: dict, open_all: bool = False
    ):
        self.changes, self.expanded, self.tabs = changes, expanded, tabs
        self.open_all = open_all
        self.items: list[_Item] = []
        self.scopes = [_Scope((), False)]

    def key(self, kind: str, name: str = "") -> tuple:
        scope = self.scopes[-1]
        occurrence = scope.occurrences.get((kind, name), 0)
        scope.occurrences[kind, name] = occurrence + 1
        return (*scope.path, (kind, name, occurrence))

    def record(self, kind: str, name: str, value, desc: str = "", **details) -> object:
        key = self.key(kind, name)
        change = self.changes.pop(key, None)
        self.items.append(
            _Item(kind, key, name, value if change is None else change, desc, details)
        )
        return change

    def begin_child(
        self, name: str, label: str, flags: merian.Properties.ChildFlags
    ) -> bool:
        scope = self.scopes[-1]
        path = self.key("child", name)
        if scope.holds_tabs:
            scope.first_tab = scope.first_tab or path
            opened = self.tabs.get(scope.path, scope.first_tab) == path
        elif flags & self.ChildFlags.TABS:
            opened = True
        else:
            default = len(self.scopes) == 1 or bool(
                flags & self.ChildFlags.DEFAULT_OPEN
            )
            opened = self.open_all or self.expanded.get(path, default)
        # A change to apply in it
        opened = opened or any(key[: len(path)] == path for key in self.changes)
        self.items.append(
            _Item("child", path, label, details={"flags": flags, "opened": opened})
        )
        if opened:
            self.scopes.append(_Scope(path, bool(flags & self.ChildFlags.TABS)))
        return opened

    def end_child(self) -> None:
        self.items.append(_Item("end", self.scopes.pop().path))

    def separate(self, label: str) -> None:
        self.items.append(_Item("separator", self.key("separator", label), label))

    def output_text(self, text: str) -> None:
        self.items.append(_Item("output", self.key("output"), value=text))

    def output_plot(
        self,
        label: str,
        samples: list[float],
        scale_min: float | None,
        scale_max: float | None,
    ) -> None:
        value = (samples, scale_min, scale_max)
        self.items.append(_Item("plot", self.key("plot", label), label, value))

    def config_float(
        self,
        name: str,
        value: float | tuple[float, ...],
        desc: str,
        sensitivity: float,
        minimum: float | None,
        maximum: float | None,
    ) -> float | tuple[float, ...] | None:
        return self.record(
            "float",
            name,
            value,
            desc,
            step=sensitivity,
            minimum=minimum,
            maximum=maximum,
        )

    def config_int(
        self,
        name: str,
        value: int | tuple[int, ...],
        desc: str,
        minimum: int,
        maximum: int,
    ) -> int | tuple[int, ...] | None:
        return self.record(
            "int", name, value, desc, step=1, minimum=minimum, maximum=maximum
        )

    def config_bool(self, name: str, value: bool, desc: str) -> bool | None:
        return self.record("bool", name, value, desc)

    def config_button(self, name: str, desc: str) -> bool:
        return bool(self.record("button", name, False, desc))

    def config_options(
        self,
        name: str,
        selected: int,
        options: list[str],
        style: merian.Properties.OptionsStyle,
        desc: str,
    ) -> int | None:
        return self.record(
            "options", name, selected, desc, options=options, style=style
        )

    def config_color(
        self, name: str, color: tuple[float, ...], desc: str
    ) -> tuple[float, ...] | None:
        return self.record("color", name, color, desc)

    def config_text(
        self, name: str, text: str, multiline: bool, needs_submit: bool, desc: str
    ) -> str | None:
        return self.record(
            "multiline" if multiline else "text",
            name,
            text,
            desc,
            needs_submit=needs_submit,
        )

    def config_path(self, name: str, path: str, desc: str) -> str | None:
        return self.record("path", name, path, desc)


# Widgets
#########


def _editing(widget: QtWidgets.QWidget) -> bool:
    focus = QtWidgets.QApplication.focusWidget()
    return focus is not None and (focus is widget or widget.isAncestorOf(focus))


def _display(label: str) -> str:
    return label.split("##")[0]


def _discard(widget: QtWidgets.QWidget) -> None:
    widget.hide()
    widget.deleteLater()


class _Section:
    def __init__(self, layout: QtWidgets.QBoxLayout):
        self.layout = layout
        self.layout.setSpacing(_SPACING)
        self.rows: dict[tuple, QtWidgets.QWidget] = {}
        self.order: list[tuple] = []
        self.cursor = 0

    def place(
        self, key: tuple, create: Callable[[], QtWidgets.QWidget]
    ) -> QtWidgets.QWidget:
        row = self.rows.get(key)
        if row is None:
            row = self.rows[key] = create()
            self.order.insert(self.cursor, key)
            self.layout.insertWidget(self.cursor, row)
        elif self.order[self.cursor] != key:
            # Rows before it were not recorded: to the end, where finish removes them
            index = self.order.index(key, self.cursor)
            skipped = self.order[self.cursor : index]
            del self.order[self.cursor : index]
            for skipped_key in skipped:
                self.layout.removeWidget(self.rows[skipped_key])
            for skipped_key in skipped:
                self.layout.insertWidget(len(self.order), self.rows[skipped_key])
                self.order.append(skipped_key)
        self.cursor += 1
        return row

    def finish(self) -> None:
        for key in self.order[self.cursor :]:
            row = self.rows.pop(key)
            self.layout.removeWidget(row)
            _discard(row)
        del self.order[self.cursor :]
        self.cursor = 0


class _Title(QtWidgets.QWidget):
    """A node's title, recorded as "[STATE] identifier (type)"."""

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(4, 4, 4, 10)
        layout.setVerticalSpacing(0)
        self.name = QtWidgets.QLabel()
        font = self.name.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.3)
        self.name.setFont(font)
        self.node_type = QtWidgets.QLabel()
        self.node_type.setObjectName("propertyLabel")
        self.state = QtWidgets.QLabel()
        layout.addWidget(self.name, 0, 0)
        layout.addWidget(self.state, 0, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.node_type, 1, 0, 1, 2)
        layout.setColumnStretch(0, 1)
        self.label = ""

    def set_label(self, label: str) -> None:
        if label == self.label:
            return
        self.label = label
        state, separator, rest = label.partition("] ")
        if not (label.startswith("[") and separator):
            state, rest = "", label
        name, _, node_type = rest.partition(" (")
        self.name.setText(name)
        self.node_type.setText(node_type.removesuffix(")"))
        self.state.setText(state.removeprefix("["))
        self.state.setVisible(bool(state))
        text = self.state.text()
        state = _TITLE_STATES.get(text, mg.NodeState.__members__.get(text))
        color = THEME[NODE_STATE_COLORS.get(state, "text_dim")]
        self.state.setStyleSheet(
            f"color: {color}; border: 1px solid {color}; border-radius: 9px;"
            "padding: 1px 8px; font-weight: bold;"
        )


class _Guided(QtWidgets.QWidget):
    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QColor(THEME["surface_hi"]))
        x = _INDENT // 2 - 1
        painter.drawLine(x, 2, x, self.height() - 2)


class _Group(QtWidgets.QWidget):
    def __init__(
        self,
        label: str,
        expanded: bool,
        toggled: Callable[[bool], None],
        titled: bool,
    ):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(_SPACING)
        self.label = label
        self.expanded = True
        self.title: _Title | None = None
        self.header: QtWidgets.QToolButton | None = None
        if titled:
            self.title = _Title()
            self.title.set_label(label)
            layout.addWidget(self.title)
            self.body = QtWidgets.QWidget()
        else:
            self.header = QtWidgets.QToolButton()
            self.header.setObjectName("groupHeader")
            self.header.setText(label)
            self.header.setToolButtonStyle(
                QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            self.header.setCheckable(True)
            self.header.toggled.connect(self.expand)
            self.header.toggled.connect(toggled)
            layout.addWidget(self.header)
            self.body = _Guided()
        body_layout = QtWidgets.QVBoxLayout(self.body)
        body_layout.setContentsMargins(0 if titled else _INDENT, 0, 0, 4)
        layout.addWidget(self.body)
        self.section = _Section(body_layout)
        self.set_expanded(expanded or titled)

    def set_label(self, label: str) -> None:
        self.label = label
        if self.title is not None:
            self.title.set_label(label)
        elif self.header.text() != label:
            self.header.setText(label)

    def set_expanded(self, expanded: bool) -> None:
        if self.header is not None:
            with QtCore.QSignalBlocker(self.header):
                self.header.setChecked(expanded)
        self.expand(expanded)

    def expand(self, expanded: bool) -> None:
        self.expanded = expanded
        self.body.setVisible(expanded)
        if self.header is not None:
            arrow = (
                QtCore.Qt.ArrowType.DownArrow
                if expanded
                else QtCore.Qt.ArrowType.RightArrow
            )
            self.header.setArrowType(arrow)


class _Page(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.section = _Section(QtWidgets.QVBoxLayout(self))
        self.section.layout.addStretch()


class _Tabs(QtWidgets.QWidget):
    def __init__(self, path: tuple, selected: Callable[[tuple, tuple], None]):
        super().__init__()
        self.path = path
        self.has_current = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        rows = QtWidgets.QWidget()
        self.section = _Section(QtWidgets.QVBoxLayout(rows))
        self.section.layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(
            lambda index: selected(self.path, (*self.path, self.order[index]))
        )
        layout.addWidget(rows)
        layout.addWidget(self.tabs)
        self.pages: dict[tuple, _Page] = {}
        self.order: list[tuple] = []
        self.cursor = 0

    def place(
        self, key: tuple, create: Callable[[], QtWidgets.QWidget]
    ) -> QtWidgets.QWidget:
        return self.section.place(key, create)

    def place_tab(self, key: tuple, label: str, current: bool) -> _Page:
        page = self.pages.get(key)
        if page is None:
            page = self.pages[key] = _Page()
        with QtCore.QSignalBlocker(self.tabs):
            if self.cursor >= len(self.order) or self.order[self.cursor] != key:
                if key in self.order:
                    self.order.remove(key)
                    self.tabs.removeTab(self.tabs.indexOf(page))
                self.order.insert(self.cursor, key)
                self.tabs.insertTab(self.cursor, page, label)
            if self.tabs.tabText(self.cursor) != label:
                self.tabs.setTabText(self.cursor, label)
            if current:
                self.tabs.setCurrentIndex(self.cursor)
                self.has_current = True
        self.cursor += 1
        return page

    def finish(self) -> None:
        self.section.finish()
        with QtCore.QSignalBlocker(self.tabs):
            for key in self.order[self.cursor :]:
                page = self.pages.pop(key)
                self.tabs.removeTab(self.tabs.indexOf(page))
                _discard(page)
        del self.order[self.cursor :]
        self.cursor = 0


def _set_text(value: object) -> str:
    """Returns value as --set takes it."""
    if isinstance(value, str):
        return value
    return json.dumps(list(value) if isinstance(value, tuple) else value)


class _LabeledRow(QtWidgets.QWidget):
    def __init__(self, label: str, field: QtWidgets.QWidget, pointer: str | None):
        super().__init__()
        self.setObjectName("propertyRow")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 2, 1)
        layout.setSpacing(10)
        self.label = QtWidgets.QLabel(label)
        self.label.setObjectName("propertyLabel")
        layout.addWidget(self.label, 2)
        if isinstance(field, _Switch):
            layout.addWidget(field, 3, QtCore.Qt.AlignmentFlag.AlignLeft)
        else:
            layout.addWidget(field, 3)
        self.field = field
        self.pointer = pointer
        self.value: object = None
        self.scrub: tuple[float, float] | None = None
        if isinstance(field, _Numbers) and field.scalar:
            self.label.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
            self.label.installEventFilter(self)

    def describe(self, desc: str) -> None:
        if self.label.toolTip() != desc:
            self.label.setToolTip(desc)
            self.field.setToolTip(desc)

    def eventFilter(self, _watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        kind = event.type()
        Type = QtCore.QEvent.Type
        box = self.field.boxes[0]
        if (
            kind == Type.MouseButtonPress
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            self.scrub = (event.globalPosition().x(), box.value())
            return True
        if kind == Type.MouseMove and self.scrub:
            start, value = self.scrub
            step = self.field.step or 1.0
            moved = event.globalPosition().x() - start
            box.setValue(
                value + round(moved / 4) * step
                if self.field.integer
                else box.shown(value + moved * step)
            )
            return True
        if kind == Type.MouseButtonRelease and self.scrub:
            self.scrub = None
            return True
        return False

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        if self.pointer is None:
            return
        clipboard = QtWidgets.QApplication.clipboard()
        menu = QtWidgets.QMenu(self)
        menu.addAction("Copy JSON Pointer", lambda: clipboard.setText(self.pointer))
        menu.addAction(
            "Copy --set Argument",
            lambda: clipboard.setText(
                f'--set "{self.pointer}={_set_text(self.value)}"'
            ),
        )
        menu.exec(event.globalPos())
        menu.deleteLater()


class _Switch(QtWidgets.QCheckBox):
    def __init__(self):
        super().__init__()
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(34, 18)

    def hitButton(self, position: QtCore.QPoint) -> bool:
        return self.rect().contains(position)

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        track = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)
        radius = track.height() / 2
        on = self.isChecked()
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(THEME["accent" if on else "surface_hi"]))
        if not self.isEnabled():
            painter.setOpacity(0.5)
        painter.drawRoundedRect(track, radius, radius)
        knob = QtCore.QRectF(0, 0, track.height() - 4, track.height() - 4)
        knob.moveCenter(
            QtCore.QPointF(
                track.right() - radius if on else track.left() + radius,
                track.center().y(),
            )
        )
        painter.setBrush(QtGui.QColor(THEME["bg" if on else "text_dim"]))
        painter.drawEllipse(knob)


class _Rule(QtWidgets.QWidget):
    def __init__(self, label: str):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 8 if label else 4, 4, 2 if label else 4)
        layout.setSpacing(8)
        if label:
            title = QtWidgets.QLabel(label)
            title.setObjectName("ruleLabel")
            layout.addWidget(title)
        line = QtWidgets.QFrame()
        line.setObjectName("rule")
        line.setFixedHeight(1)
        layout.addWidget(line, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)


def _decimals(step: float) -> int:
    if step <= 0:
        return _DEFAULT_DECIMALS
    # The tolerance keeps float32 steps such as 0.01 at their decimals
    return max(1, math.ceil(-math.log10(step) - 1e-6))


class _FocusedWheel:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _QuietCombo(_FocusedWheel, QtWidgets.QComboBox):
    pass


class _QuietSlider(_FocusedWheel, QtWidgets.QSlider):
    pass


class _NumberBox(_FocusedWheel, QtWidgets.QDoubleSpinBox):
    def __init__(self, integer: bool):
        super().__init__()
        self.integer = integer
        self.shown_decimals = 0 if integer else _DEFAULT_DECIMALS
        self.setDecimals(0 if integer else 9)
        self.setKeyboardTracking(False)
        # Unbounded ranges would size the box for their widest text
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.setMinimumWidth(56)

    def set_step(self, step: float) -> None:
        self.setSingleStep(step)
        decimals = 0 if self.integer else _decimals(step)
        if decimals != self.shown_decimals:
            self.shown_decimals = decimals
            # The box keeps its text until the value changes
            line = self.lineEdit()
            with QtCore.QSignalBlocker(line):
                line.setText(self.textFromValue(self.value()))

    def shown(self, value: float) -> float:
        return round(value, self.shown_decimals) + 0.0

    def textFromValue(self, value: float) -> str:
        if self.integer:
            return str(round(value))
        return f"{self.shown(value):.{self.shown_decimals}f}"

    def stepBy(self, steps: int) -> None:
        self.interpretText()
        self.setValue(self.shown(self.value() + steps * self.singleStep()))

    def valueFromText(self, text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, position: int):
        return QtGui.QValidator.State.Acceptable, text, position


class _Numbers(QtWidgets.QWidget):
    def __init__(self, count: int, integer: bool, commit: Callable[[object], None]):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.integer = integer
        self.scalar = count == 0
        self.step: float | None = None
        self.bounds = (-math.inf, math.inf)
        self.boxes = [_NumberBox(integer) for _ in range(max(count, 1))]
        for box in self.boxes:
            box.valueChanged.connect(lambda _: commit(self.values()))
            layout.addWidget(box)

    def values(self):
        values = tuple(box.value() for box in self.boxes)
        if self.integer:
            # Exact, beyond what the boxes' doubles hold
            low, high = self.bounds
            values = tuple(min(max(round(value), low), high) for value in values)
        return values[0] if self.scalar else values

    def configure(
        self, step: float, minimum: float | None, maximum: float | None
    ) -> None:
        bounds = (
            -math.inf if minimum is None else minimum,
            math.inf if maximum is None else maximum,
        )
        if (step, bounds) != (self.step, self.bounds):
            self.step, self.bounds = step, bounds
            for box in self.boxes:
                with QtCore.QSignalBlocker(box):
                    box.setRange(max(bounds[0], -1e18), min(bounds[1], 1e18))
                    box.set_step(step)

    def set_values(self, values) -> None:
        for box, value in zip(self.boxes, (values,) if self.scalar else values):
            if not _editing(box) and box.value() != value:
                with QtCore.QSignalBlocker(box):
                    box.setValue(value)


class _Slider(_Numbers):
    def __init__(self, integer: bool, commit: Callable[[object], None]):
        super().__init__(0, integer, commit)
        self.steps = _MAX_SLIDER_STEPS
        self.box = self.boxes[0]
        self.box.setFixedWidth(80)
        self.box.valueChanged.connect(lambda _: self.follow())
        self.slider = _QuietSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.slid)
        layout = self.layout()
        layout.setSpacing(8)
        layout.insertWidget(0, self.slider, 1)

    def slid(self, position: int) -> None:
        low, high = self.bounds
        value = (
            position
            if self.integer
            else self.box.shown(low + position / self.steps * (high - low))
        )
        if self.box.value() != value:
            self.box.setValue(value)

    def follow(self) -> None:
        low, high = self.bounds
        value = self.box.value()
        position = (
            round(value)
            if self.integer
            else round((value - low) / ((high - low) or 1) * self.steps)
        )
        with QtCore.QSignalBlocker(self.slider):
            self.slider.setValue(position)

    def configure(self, step: float, minimum: float, maximum: float) -> None:
        if (step, (minimum, maximum)) == (self.step, self.bounds):
            return
        super().configure(step, minimum, maximum)
        with QtCore.QSignalBlocker(self.slider):
            if self.integer:
                self.slider.setRange(round(minimum), round(maximum))
            else:
                span = (maximum - minimum) / step if step > 0 else _MAX_SLIDER_STEPS
                self.steps = max(1, min(round(span), _MAX_SLIDER_STEPS))
                self.slider.setRange(0, self.steps)
        self.follow()

    def set_values(self, value) -> None:
        if not _editing(self) and self.box.value() != value:
            super().set_values(value)
            self.follow()


class _Segments(QtWidgets.QWidget):
    activated = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.row = QtWidgets.QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(0)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.idClicked.connect(self.activated)
        self.options: list[str] = []

    def set_options(self, options: list[str]) -> None:
        if options == self.options:
            return
        self.options = list(options)
        for button in self.group.buttons():
            self.group.removeButton(button)
            _discard(button)
        for index, option in enumerate(options):
            button = QtWidgets.QToolButton(text=option, checkable=True)
            button.setObjectName("segment")
            first, last = index == 0, index == len(options) - 1
            button.setProperty(
                "segment",
                "only"
                if first and last
                else "first"
                if first
                else "last"
                if last
                else "middle",
            )
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            self.group.addButton(button, index)
            self.row.addWidget(button)

    def set_current(self, index: int) -> None:
        if (button := self.group.button(index)) and not button.isChecked():
            button.setChecked(True)


def _sliding(details: dict) -> bool:
    low, high = details["minimum"], details["maximum"]
    if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)):
        return False
    return 0 < high - low <= _MAX_SLIDER_SPAN


def _segmented(options: list[str], style: merian.Properties.OptionsStyle) -> bool:
    styles = merian.Properties.OptionsStyle
    if style == styles.COMBO:
        return False
    return style == styles.RADIO_BUTTON or (
        len(options) <= _MAX_SEGMENTS
        and all(len(option) <= _MAX_SEGMENT_LENGTH for option in options)
    )


class _MultilineText(QtWidgets.QPlainTextEdit):
    submitted = QtCore.Signal(str)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.submitted.emit(self.toPlainText())


class _PathEdit(QtWidgets.QWidget):
    def __init__(self, commit: Callable[[str], None]):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.line = QtWidgets.QLineEdit()
        self.line.editingFinished.connect(lambda: commit(self.line.text()))
        browse = QtWidgets.QToolButton()
        browse.setText("\u2026")
        browse.clicked.connect(lambda: self.browse(commit))
        layout.addWidget(self.line)
        layout.addWidget(browse)

    def browse(self, commit: Callable[[str], None]) -> None:
        directory = str(Path(self.line.text()).parent) if self.line.text() else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open", directory)
        if path:
            self.line.setText(path)
            commit(path)

    def set_path(self, path: str) -> None:
        if not _editing(self.line) and self.line.text() != path:
            self.line.setText(path)


def _to_qcolor(color: tuple[float, ...]) -> QtGui.QColor:
    return QtGui.QColor.fromRgbF(
        *(min(max(component, 0.0), 1.0) for component in color)
    )


class _ColorEdit(QtWidgets.QWidget):
    def __init__(self, count: int, commit: Callable[[tuple], None]):
        super().__init__()
        self.commit = commit
        self.color: tuple[float, ...] = ()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.swatch = QtWidgets.QPushButton()
        self.swatch.setFixedSize(34, 24)
        self.swatch.setToolTip("Pick a color")
        self.swatch.clicked.connect(self.pick)
        layout.addWidget(self.swatch)
        self.components = _Numbers(count, False, commit)
        self.components.configure(0.01, 0.0, None)
        layout.addWidget(self.components, 1)

    def pick(self) -> None:
        dialog = QtWidgets.QColorDialog(_to_qcolor(self.color), self)
        dialog.setOption(
            QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel,
            len(self.color) == 4,
        )
        if dialog.exec():
            self.commit(dialog.selectedColor().getRgbF()[: len(self.color)])
        dialog.deleteLater()

    def set_color(self, color: tuple[float, ...]) -> None:
        if color != self.color:
            self.color = color
            self.swatch.setStyleSheet(
                f"background-color: {_to_qcolor(color[:3]).name()};"
                f"border: 1px solid {THEME['surface_hi']}; border-radius: 6px;"
            )
        self.components.set_values(color)


class _Plot(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.samples: list[float] = []
        self.scale = (0.0, 1.0)
        self.setMinimumHeight(60)

    def set_samples(
        self, samples: list[float], scale_min: float | None, scale_max: float | None
    ) -> None:
        finite = [sample for sample in samples if math.isfinite(sample)] or [0.0]
        self.samples = samples
        self.scale = (
            min(finite) if scale_min is None else scale_min,
            max(finite) if scale_max is None else scale_max,
        )
        self.update()

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        frame = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(THEME["surface"]))
        painter.drawRoundedRect(frame, 6, 6)
        if len(self.samples) < 2:
            return
        low, high = self.scale
        span = high - low or 1.0
        area = frame.adjusted(4, 6, -4, -4)
        points = [
            QtCore.QPointF(
                area.left() + area.width() * i / (len(self.samples) - 1),
                area.top()
                + area.height() * (1 - (min(max(sample, low), high) - low) / span),
            )
            for i, sample in enumerate(self.samples)
            if math.isfinite(sample)
        ]
        if not points:
            return
        fill = QtGui.QPainterPath(QtCore.QPointF(points[0].x(), area.bottom()))
        for point in points:
            fill.lineTo(point)
        fill.lineTo(points[-1].x(), area.bottom())
        color = QtGui.QColor(THEME["accent"])
        color.setAlpha(50)
        painter.fillPath(fill, color)
        painter.setPen(QtGui.QPen(QtGui.QColor(THEME["accent"]), 1.5))
        painter.drawPolyline(points)


# Panel
#######


class QtProperties:
    """
    Shows recordings in widgets inside a scroll area (`widget`) and keeps what the user
    changes, expands or selects for the next recorder(), which `edited` asks for.
    """

    def __init__(
        self,
        edited: Callable[[], None],
        pointer_of: Callable[[tuple], str | None],
    ):
        self.edited = edited
        self.pointer_of = pointer_of
        self.filter = ""
        self.widget = QtWidgets.QWidget()
        self.search = QtWidgets.QLineEdit(placeholderText="Filter properties")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.set_filter)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(8, 4, 8, 8)
        self.root = _Section(layout)
        layout.addStretch()
        panel = QtWidgets.QVBoxLayout(self.widget)
        panel.setContentsMargins(8, 8, 0, 0)
        panel.addWidget(self.search)
        panel.addWidget(scrolled(content), 1)
        # {key: value} not yet handed to a recorder, and those handed but not shown back
        self.changes: dict[tuple, object] = {}
        self.sent: dict[tuple, object] = {}
        self.expanded: dict[tuple, bool] = {}
        self.tabs: dict[tuple, tuple] = {}

    def recorder(self) -> Recorder:
        self.sent |= self.changes
        changes, self.changes = self.changes, {}
        return Recorder(
            changes, dict(self.expanded), dict(self.tabs), bool(self.filter)
        )

    def set_filter(self, text: str) -> None:
        self.filter = text.strip().lower()
        _filter(self.root, self.filter)
        self.edited()

    def clear(self) -> None:
        self.root.finish()
        self.changes.clear()
        self.sent.clear()
        self.tabs.clear()

    def commit(self, key: tuple) -> Callable[[object], None]:
        def set_change(value: object) -> None:
            self.changes[key] = value
            self.edited()

        return set_change

    def toggle(self, path: tuple, expanded: bool) -> None:
        self.expanded[path] = expanded
        self.edited()

    def select_tab(self, path: tuple, tab: tuple) -> None:
        self.tabs[path] = tab
        self.edited()

    def show(self, items: list[_Item], consumed: bool) -> None:
        """Renders a recording; consumed: its recorder took all changes sent so far."""
        if consumed:
            self.sent.clear()
        containers: list[_Section | _Tabs] = [self.root]
        for item in items:
            if item.kind == "end":
                container = containers.pop()
                if isinstance(container, _Tabs) and not container.has_current:
                    if self.tabs.pop(container.path, None) is not None:
                        self.edited()
                container.finish()
            elif item.kind == "child":
                if entered := self.show_child(containers[-1], item):
                    containers.append(entered)
            else:
                self.show_row(containers[-1], item)
        for container in reversed(containers):
            container.finish()
        _filter(self.root, self.filter)

    def show_child(
        self, container: _Section | _Tabs, item: _Item
    ) -> _Section | _Tabs | None:
        opened, flags = item.details["opened"], item.details["flags"]
        if isinstance(container, _Tabs):
            # The panel's own selection over a recording made before it
            current = (
                self.tabs[container.path] == item.key
                if container.path in self.tabs
                else opened
            )
            page = container.place_tab(item.key[-1], _display(item.label), current)
            return page.section if opened else None

        def create():
            if flags & merian.Properties.ChildFlags.TABS:
                return _Tabs(item.key, self.select_tab)
            return _Group(
                _display(item.label),
                opened,
                functools.partial(self.toggle, item.key),
                titled=len(item.key) == 1,
            )

        child = container.place(item.key[-1], create)
        if isinstance(child, _Tabs):
            child.has_current = False
            return child
        child.set_label(_display(item.label))
        child.set_expanded(self.expanded.get(item.key, opened))
        return child.section if opened else None

    def show_row(self, container: _Section | _Tabs, item: _Item) -> None:
        key = item.key[-1]
        pending = item.key in self.changes or item.key in self.sent
        create = functools.partial(self.create_row, item)
        row = container.place(key, create)
        if isinstance(row, _LabeledRow):
            row.describe(item.desc)
            row.value = (
                item.details["options"][item.value]
                if item.kind == "options"
                else item.value
            )
            widget = row.field
        else:
            widget = row
        if not pending:
            self.update_row(item, widget)

    def create_row(self, item: _Item) -> QtWidgets.QWidget:
        commit = self.commit(item.key)
        kind = item.kind
        if kind == "separator":
            return _Rule(_display(item.label))
        if kind == "output":
            output = QtWidgets.QLabel()
            output.setObjectName("note")
            output.setWordWrap(True)
            output.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            return output
        if kind == "button":
            button = QtWidgets.QPushButton(_display(item.label))
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
            )
            button.clicked.connect(lambda: commit(True))
            return button
        if kind == "plot":
            field = _Plot()
        elif kind in ("float", "int"):
            count = len(item.value) if isinstance(item.value, tuple) else 0
            if count == 0 and _sliding(item.details):
                field = _Slider(kind == "int", commit)
            else:
                field = _Numbers(count, kind == "int", commit)
        elif kind == "bool":
            field = _Switch()
            field.toggled.connect(commit)
        elif kind == "options":
            if _segmented(item.details["options"], item.details["style"]):
                field = _Segments()
            else:
                field = _QuietCombo()
            field.activated.connect(commit)
        elif kind == "color":
            field = _ColorEdit(len(item.value), commit)
        elif kind == "path":
            field = _PathEdit(commit)
        elif kind == "multiline":
            field = _MultilineText()
            if item.details["needs_submit"]:
                field.submitted.connect(commit)
            else:
                field.textChanged.connect(lambda: commit(field.toPlainText()))
        else:
            field = QtWidgets.QLineEdit()
            if item.details["needs_submit"]:
                field.editingFinished.connect(lambda: commit(field.text()))
            else:
                field.textEdited.connect(commit)
        pointer = None if kind == "plot" else self.pointer_of(item.key)
        return _LabeledRow(_display(item.label), field, pointer)

    def update_row(self, item: _Item, field: QtWidgets.QWidget) -> None:
        kind, value = item.kind, item.value
        if kind == "output":
            if field.text() != value:
                field.setText(value)
                color = next(
                    (
                        THEME[color]
                        for prefix, color in _NOTE_COLORS.items()
                        if value.startswith(prefix)
                    ),
                    None,
                )
                field.setStyleSheet(
                    f"color: {color}; border-left: 3px solid {color};"
                    "padding: 4px 8px; margin: 2px 0;"
                    if color
                    else ""
                )
        elif kind == "button":
            field.setToolTip(item.desc)
        elif kind == "plot":
            field.set_samples(*value)
        elif kind in ("float", "int"):
            field.configure(
                item.details["step"], item.details["minimum"], item.details["maximum"]
            )
            field.set_values(value)
        elif kind == "bool":
            if field.isChecked() != value:
                with QtCore.QSignalBlocker(field):
                    field.setChecked(value)
        elif kind == "options" and isinstance(field, _Segments):
            field.set_options(item.details["options"])
            field.set_current(value)
        elif kind == "options":
            with QtCore.QSignalBlocker(field):
                options = item.details["options"]
                if [field.itemText(index) for index in range(field.count())] != options:
                    field.clear()
                    field.addItems(options)
                if field.currentIndex() != value:
                    field.setCurrentIndex(value)
        elif kind == "color":
            field.set_color(value)
        elif kind == "path":
            field.set_path(value)
        elif kind in ("text", "multiline"):
            current = field.toPlainText() if kind == "multiline" else field.text()
            if not _editing(field) and current != value:
                with QtCore.QSignalBlocker(field):
                    if kind == "multiline":
                        field.setPlainText(value)
                    else:
                        field.setText(value)


def _filter(section: _Section, text: str) -> bool:
    shown = False
    for row in section.rows.values():
        if isinstance(row, _Group):
            matched = bool(text) and text in row.label.lower()
            visible = _filter(row.section, "" if matched else text) or matched
            row.body.setVisible(row.expanded or bool(text))
            visible = visible or not text or row.title is not None
        elif isinstance(row, _Tabs):
            visible = _filter(row.section, text)
            for page in row.pages.values():
                visible = _filter(page.section, text) or visible
            visible = visible or not text
        elif isinstance(row, _LabeledRow):
            visible = not text or text in row.label.text().lower()
        else:
            visible = not text
        row.setVisible(visible)
        shown = shown or visible
    return shown
