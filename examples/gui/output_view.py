import enum
import functools
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import merian.graph as mg

from .theme import RIGHT_ALIGNED
from .window_input import WindowInput

_BUFFER_HEIGHT = 420
_COMPONENT_NAMES = "xyzw"
_DEFAULT_BYTES = 2**20
_MAX_BYTES = 2**31 - 1
_NO_SOURCE = "None"
_RESIZE_DELAY = timedelta(milliseconds=100)
_SHOWN_AS = ("values", "hex", "bits", "colors", "image")
_SPACING = 4
_TYPES = (
    "float32",
    "float16",
    "float64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
)


def _normalized(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        return (values.astype(np.float64) - info.min) / (float(info.max) - info.min)
    return np.clip(np.nan_to_num(values.astype(np.float64)), 0, 1)


def _to_rgb(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] == 1:
        return np.repeat(values, 3, axis=-1)
    padding = [(0, 0)] * (values.ndim - 1) + [(0, max(3 - values.shape[-1], 0))]
    return np.pad(values[..., :3], padding)


def _to_color(components: np.ndarray) -> QtGui.QColor:
    return QtGui.QColor.fromRgbF(*_to_rgb(_normalized(components)))


def _display_texels(texels: np.ndarray, texel_format: str) -> np.ndarray:
    if texels.ndim == 4:
        texels = texels[0]
    if texels.dtype == np.uint8 and texels.shape[-1] == 4:
        return (
            texels[..., [2, 1, 0, 3]] if texel_format.startswith("B8G8R8A8") else texels
        )
    rgb = _to_rgb(_normalized(texels))
    if np.issubdtype(texels.dtype, np.floating):
        rgb = np.where(
            rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055
        )
    display = np.full((*rgb.shape[:2], 4), 255, np.uint8)
    display[..., :3] = np.round(rgb * 255)
    return display


class SourceKind(enum.Enum):
    OUTPUT = enum.auto()
    WINDOW = enum.auto()
    SINK = enum.auto()


@dataclass(frozen=True)
class Source:
    kind: SourceKind
    identifier: str
    output: str = ""
    content: mg.ConnectorKind = mg.ConnectorKind.IMAGE
    title: str = field(default="", compare=False)

    def label(self) -> str:
        if self.kind == SourceKind.OUTPUT:
            return f"{self.identifier}.{self.output}"
        if self.kind == SourceKind.WINDOW:
            return self.title
        return f"{self.identifier} (sink)"


def _centered(size: QtCore.QSize, rect: QtCore.QRect) -> QtCore.QRect:
    target = QtCore.QRect(QtCore.QPoint(), size)
    target.moveCenter(rect.center())
    return target


def fit_properties(widget: QtWidgets.QWidget) -> dict:
    """Returns the Host Image Sink properties that fit images to the widget."""
    ratio = widget.devicePixelRatioF()
    return {"fit": [round(widget.width() * ratio), round(widget.height() * ratio)]}


def resize_timer(
    widget: QtWidgets.QWidget, timeout: Callable[[], None]
) -> QtCore.QTimer:
    """Returns a debounce timer calling timeout; start() it on each resize."""
    return QtCore.QTimer(
        widget,
        singleShot=True,
        timeout=timeout,
        interval=_RESIZE_DELAY // timedelta(milliseconds=1),
    )


class _BufferModel(QtCore.QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.values = np.zeros((0, 1), np.float32)
        self.bits = self.values.view(np.uint32)
        self.shown_as = "values"
        self.first = 0

    def set_values(self, values: np.ndarray, shown_as: str, first: int) -> None:
        reset = values.shape != self.values.shape or values.dtype != self.values.dtype
        renumbered = first != self.first
        if reset:
            self.beginResetModel()
        self.values, self.shown_as, self.first = values, shown_as, first
        self.bits = values.view(np.dtype(f"u{values.dtype.itemsize}"))
        if reset:
            self.endResetModel()
        elif len(values):
            self.dataChanged.emit(
                self.index(0, 0), self.index(len(values) - 1, values.shape[1] - 1)
            )
            if renumbered:
                self.headerDataChanged.emit(
                    QtCore.Qt.Orientation.Vertical, 0, len(values) - 1
                )

    def rowCount(self, _parent=None) -> int:
        return len(self.values)

    def columnCount(self, _parent=None) -> int:
        return self.values.shape[1]

    def data(self, index: QtCore.QModelIndex, role=QtCore.Qt.ItemDataRole.DisplayRole):
        row, column = index.row(), index.column()
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if self.shown_as == "hex":
                return f"{int(self.bits[row, column]):0{2 * self.values.itemsize}x}"
            if self.shown_as == "bits":
                return f"{int(self.bits[row, column]):0{8 * self.values.itemsize}b}"
            value = self.values[row, column]
            return (
                f"{value:.6g}"
                if np.issubdtype(self.values.dtype, np.floating)
                else str(value)
            )
        if role == QtCore.Qt.ItemDataRole.BackgroundRole and self.shown_as == "colors":
            return _to_color(self.values[row])
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return RIGHT_ALIGNED
        return None

    def headerData(
        self, section: int, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole
    ):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return _COMPONENT_NAMES[section] if self.values.shape[1] > 1 else "value"
        return str(self.first + section)


def _source_groups(sources: list[Source]) -> Iterator[tuple[str, list[Source]]]:
    for kind, title in (
        (SourceKind.WINDOW, "Windows"),
        (SourceKind.SINK, "Host Sinks"),
        (SourceKind.OUTPUT, "Outputs"),
    ):
        if of_kind := [source for source in sources if source.kind == kind]:
            yield title, of_kind


def _select_source_item(combo: QtWidgets.QComboBox, current: Source | None) -> None:
    index = next(
        (index for index in range(combo.count()) if combo.itemData(index) == current),
        -1,
    )
    with QtCore.QSignalBlocker(combo):
        if index < 0 and current is not None:
            combo.addItem(current.label(), current)
            index = combo.count() - 1
        combo.setCurrentIndex(max(index, 0))


def set_source_actions(
    menu: QtWidgets.QMenu,
    sources: list[Source],
    current: Source | None,
    triggered: Callable[[Source | None], None],
) -> None:
    menu.clear()
    group = menu.findChild(QtGui.QActionGroup) or QtGui.QActionGroup(menu)

    def add(label: str, source: Source | None) -> None:
        action = menu.addAction(label)
        action.setCheckable(True)
        action.setChecked(source == current)
        action.triggered.connect(lambda: triggered(source))
        group.addAction(action)

    add(_NO_SOURCE, None)
    for title, of_kind in _source_groups(sources):
        menu.addSeparator()
        menu.addAction(title).setEnabled(False)
        for source in of_kind:
            add(source.label(), source)
    if current is not None and current not in sources:
        add(current.label(), current)


class CaptureImage:
    def __init__(self):
        self.image: QtGui.QImage | None = None
        # Owns the memory image refers to
        self.texels = None
        self.drawn = QtCore.QRectF()

    def area(self, rect: QtCore.QRect) -> QtCore.QRectF:
        return self.drawn if self.image is not None else QtCore.QRectF(rect)

    def set_capture(
        self, capture: mg.Capture | mg.HostWindow.Frame | None, pixel_ratio: float
    ) -> None:
        self.texels = (
            _display_texels(capture.array, capture.format) if capture else None
        )
        self.image = None
        if self.texels is not None:
            height, width, _ = self.texels.shape
            self.image = QtGui.QImage(
                self.texels.data,
                width,
                height,
                self.texels.strides[0],
                QtGui.QImage.Format.Format_RGBX8888,
            )
            self.image.setDevicePixelRatio(pixel_ratio)

    def draw(self, painter: QtGui.QPainter, rect: QtCore.QRect) -> None:
        size = self.image.deviceIndependentSize().toSize()
        if size.width() > rect.width() or size.height() > rect.height():
            size = size.scaled(rect.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        target = _centered(size, rect)
        self.drawn = QtCore.QRectF(target)
        painter.drawImage(target, self.image)


class _ImagePage(QtWidgets.QWidget):
    sink_changed = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self.capture_image = CaptureImage()
        self.setMinimumSize(160, 90)
        self.resize_timer = resize_timer(
            self, lambda: self.sink_changed.emit(self.sink_properties())
        )
        self.input = WindowInput(self, lambda: self.capture_image.area(self.rect()))

    def sink_properties(self) -> dict:
        return fit_properties(self)

    def aspect(self) -> float | None:
        image = self.capture_image.image
        return image.height() / image.width() if image is not None else None

    def show_capture(self, capture: mg.Capture | None) -> None:
        self.capture_image.set_capture(capture, self.devicePixelRatioF())
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.resize_timer.start()

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        if self.capture_image.image is not None:
            self.capture_image.draw(QtGui.QPainter(self), self.rect())


class _ImageView(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.image: QtGui.QImage | None = None

    def set_image(self, image: QtGui.QImage | None) -> None:
        self.image = image
        self.update()

    def paintEvent(self, _: QtGui.QPaintEvent) -> None:
        if self.image is None:
            return
        size = self.image.size().scaled(
            self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        QtGui.QPainter(self).drawImage(_centered(size, self.rect()), self.image)


class _BufferPage(QtWidgets.QWidget):
    sink_changed = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self.raw: np.ndarray | None = None
        self.type = QtWidgets.QComboBox()
        self.type.addItems(_TYPES)
        self.components = QtWidgets.QSpinBox(minimum=1, maximum=4)
        self.shown_as = QtWidgets.QComboBox()
        self.shown_as.addItems(_SHOWN_AS)
        self.image_width = QtWidgets.QSpinBox(
            minimum=0, maximum=2**16, specialValueText="auto"
        )
        self.image_width.setToolTip("Elements per image row, auto: a square")
        self.byte_offset = QtWidgets.QSpinBox(
            minimum=0, maximum=_MAX_BYTES, singleStep=4
        )
        self.byte_offset.setToolTip("First byte captured")
        self.byte_count = QtWidgets.QSpinBox(
            minimum=0, maximum=_MAX_BYTES, singleStep=4, specialValueText="all"
        )
        self.byte_count.setValue(_DEFAULT_BYTES)
        self.byte_count.setToolTip("Bytes captured, all: to the end of the buffer")
        self.info = QtWidgets.QLabel()
        for control in (self.type, self.shown_as):
            control.currentIndexChanged.connect(self.show_values)
        for control in (self.components, self.image_width):
            control.valueChanged.connect(self.show_values)
        for control in (self.byte_offset, self.byte_count):
            control.setKeyboardTracking(False)
            control.valueChanged.connect(
                lambda: self.sink_changed.emit(self.sink_properties())
            )

        controls = QtWidgets.QGridLayout()
        for column, (label, control) in enumerate(
            (
                ("type", self.type),
                ("components", self.components),
                ("as", self.shown_as),
                ("width", self.image_width),
                ("offset", self.byte_offset),
                ("bytes", self.byte_count),
            )
        ):
            controls.addWidget(QtWidgets.QLabel(label), column // 3 * 2, column % 3)
            controls.addWidget(control, column // 3 * 2 + 1, column % 3)
        self.model = _BufferModel()
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.image = _ImageView()
        self.pages = QtWidgets.QStackedWidget()
        self.pages.addWidget(self.table)
        self.pages.addWidget(self.image)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.info)
        layout.addWidget(self.pages, 1)

    def sink_properties(self) -> dict:
        return {"offset": self.byte_offset.value(), "size": self.byte_count.value()}

    def show_capture(self, capture: mg.Capture | None) -> None:
        self.raw = capture.array if capture is not None else None
        self.show_values()

    def show_values(self) -> None:
        if self.raw is None:
            self.info.setText("waiting for a capture")
            self.model.set_values(np.zeros((0, 1), np.float32), "values", 0)
            self.image.set_image(None)
            return
        dtype = np.dtype(self.type.currentText())
        components = self.components.value()
        element_size = dtype.itemsize * components
        count = len(self.raw) // element_size
        values = self.raw[: count * element_size].view(dtype).reshape(count, components)
        shown_as = self.shown_as.currentText()
        self.info.setText(
            f"{len(self.raw)} bytes, {count} elements of {components} {dtype}"
        )
        first = self.byte_offset.value() // element_size
        if shown_as == "image":
            self.pages.setCurrentWidget(self.image)
            self.image.set_image(self.to_image(values))
        else:
            self.pages.setCurrentWidget(self.table)
            self.model.set_values(values, shown_as, first)

    def to_image(self, values: np.ndarray) -> QtGui.QImage | None:
        width = self.image_width.value() or max(1, math.isqrt(len(values)))
        height = len(values) // width
        if height == 0:
            return None
        step_y = math.ceil(height / max(self.image.height(), 1))
        step_x = math.ceil(width / max(self.image.width(), 1))
        texels = values[: width * height].reshape(height, width, -1)[::step_y, ::step_x]
        texels = texels.astype(np.float32)
        height, width = texels.shape[:2]
        finite = texels[np.isfinite(texels)]
        low, high = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        texels = np.nan_to_num((texels - low) / (high - low or 1.0))
        rgb = np.round(np.clip(_to_rgb(texels), 0, 1) * 255).astype(np.uint8)
        return QtGui.QImage(
            rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format.Format_RGB888
        ).copy()


class OutputView(QtWidgets.QWidget):
    closed = QtCore.Signal()
    # A Source or None from the header
    source_requested = QtCore.Signal(object)
    # Properties of the Host Sink that captures an output
    sink_changed = QtCore.Signal(dict)
    interval_changed = QtCore.Signal(object)
    aspect_changed = QtCore.Signal()

    def __init__(self, placeholder: str, close_tooltip: str):
        super().__init__()
        self.source: Source | None = None
        self.aspect = 9 / 16
        self.source_combo = QtWidgets.QComboBox(minimumContentsLength=14)
        self.source_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.source_combo.setToolTip("What the view shows")
        self.source_combo.activated.connect(self.request_source)
        self.interval = QtWidgets.QSpinBox(
            minimum=0,
            maximum=10000,
            singleStep=50,
            suffix=" ms",
            specialValueText="no interval",
        )
        self.interval.setToolTip(
            "Least time between captures; without, as fast as they are shown"
        )
        self.interval.valueChanged.connect(
            lambda: self.interval_changed.emit(self.capture_interval())
        )
        self.close_button = QtWidgets.QToolButton(
            text="\u00d7", autoRaise=True, toolTip=close_tooltip
        )
        self.close_button.clicked.connect(self.closed)
        self.placeholder = QtWidgets.QLabel(
            placeholder, wordWrap=True, alignment=QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.placeholder.setEnabled(False)
        self.image = _ImagePage()
        self.image.sink_changed.connect(
            functools.partial(self.emit_sink_changed, mg.ConnectorKind.IMAGE)
        )
        self.buffer = _BufferPage()
        self.buffer.sink_changed.connect(
            functools.partial(self.emit_sink_changed, mg.ConnectorKind.BUFFER)
        )
        self.pages = QtWidgets.QStackedWidget()
        for page in (self.placeholder, self.image, self.buffer):
            self.pages.addWidget(page)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(_SPACING, 0, 0, 0)
        header.addWidget(self.source_combo, 1)
        header.addWidget(self.interval)
        header.addWidget(self.close_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.pages, 1)
        self.set_source(None)

    @property
    def input(self) -> WindowInput:
        return self.image.input

    def set_sources(self, sources: list[Source]) -> None:
        with QtCore.QSignalBlocker(self.source_combo):
            self.source_combo.clear()
            self.source_combo.addItem(_NO_SOURCE, None)
            for _, of_kind in _source_groups(sources):
                self.source_combo.insertSeparator(self.source_combo.count())
                for source in of_kind:
                    self.source_combo.addItem(source.label(), source)
            _select_source_item(self.source_combo, self.source)

    def request_source(self, index: int) -> None:
        self.source_requested.emit(self.source_combo.itemData(index))
        _select_source_item(self.source_combo, self.source)

    def set_source(self, source: Source | None) -> None:
        self.source = source
        _select_source_item(self.source_combo, source)
        self.interval.setVisible(
            source is not None and source.kind != SourceKind.WINDOW
        )
        self.pages.setCurrentWidget(self.page())
        self.input.set_window(None)
        self.image.show_capture(None)
        self.buffer.show_capture(None)

    def page(self) -> QtWidgets.QWidget:
        if self.source is None:
            return self.placeholder
        if self.source.content == mg.ConnectorKind.BUFFER:
            return self.buffer
        return self.image

    def emit_sink_changed(self, content: mg.ConnectorKind, properties: dict) -> None:
        if self.source is not None and content == self.source.content:
            self.sink_changed.emit(properties)

    def sink_properties(self) -> dict:
        return self.page().sink_properties() if self.source is not None else {}

    def capture_interval(self) -> timedelta:
        return timedelta(milliseconds=self.interval.value())

    def show_capture(self, capture: mg.Capture | mg.HostWindow.Frame | None) -> None:
        if self.source is None:
            return
        self.page().show_capture(capture)
        aspect = self.image.aspect()
        if (
            self.source.content == mg.ConnectorKind.IMAGE
            and aspect
            and abs(aspect - self.aspect) > 0.01 * self.aspect
        ):
            self.aspect = aspect
            self.aspect_changed.emit()

    def height_for(self, width: int) -> int:
        content = (
            _BUFFER_HEIGHT
            if self.source and self.source.content == mg.ConnectorKind.BUFFER
            else round(width * self.aspect)
        )
        return self.source_combo.sizeHint().height() + content


class OutputPanel(QtWidgets.QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        self.entries = QtWidgets.QVBoxLayout(content)
        self.entries.addStretch()
        self.setWidget(content)

    def add(self, output_view: OutputView) -> None:
        self.entries.insertWidget(self.entries.count() - 1, output_view)
        output_view.aspect_changed.connect(self.fit_entries)
        self.fit_entries()

    def fit_entries(self) -> None:
        width = self.viewport().width()
        for index in range(self.entries.count() - 1):
            output_view = self.entries.itemAt(index).widget()
            output_view.setFixedHeight(output_view.height_for(width))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.fit_entries()
