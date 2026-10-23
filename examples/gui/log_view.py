import collections
import re
from collections.abc import Iterable
from datetime import timedelta
from pathlib import PurePath

from PySide6 import QtCore, QtGui, QtWidgets

import merian

from .theme import THEME

_LEVEL_COLORS = {
    merian.LogLevel.TRACE: "text_dim",
    merian.LogLevel.DEBUG: "teal",
    merian.LogLevel.INFO: "green",
    merian.LogLevel.WARN: "yellow",
    merian.LogLevel.ERROR: "red",
    merian.LogLevel.CRITICAL: "red",
}
_LEVEL_NAMES = {level: level.name.lower() for level in _LEVEL_COLORS}
_LEVEL_WIDTH = 5
_MIN_HEIGHT = 200
_POLL_INTERVAL = timedelta(milliseconds=100)
_RECORDS = 5000
_BLOCK_BREAKS = re.compile(r"\r?\n|[\r\u2029]")
_REFILTER_DELAY = timedelta(milliseconds=150)


def _text_format(color: str, bold: bool = False) -> QtGui.QTextCharFormat:
    text_format = QtGui.QTextCharFormat()
    text_format.setForeground(QtGui.QColor(THEME[color]))
    if bold:
        text_format.setFontWeight(QtGui.QFont.Weight.Bold)
    return text_format


class LogView(QtWidgets.QWidget):
    def __init__(self, sink: merian.LogSink):
        super().__init__()
        self.sink = sink
        self.stale = True
        self.records: collections.deque[merian.LogRecord] = collections.deque(
            maxlen=_RECORDS
        )
        self.shown: collections.deque[merian.LogRecord] = collections.deque(
            maxlen=_RECORDS
        )
        self.level_formats = {
            level: _text_format(color, level == merian.LogLevel.CRITICAL)
            for level, color in _LEVEL_COLORS.items()
        }
        self.dim_format = _text_format("text_dim")
        self.message_format = _text_format("text")

        self.level_combo = QtWidgets.QComboBox(toolTip="The least level shown")
        for level, name in _LEVEL_NAMES.items():
            self.level_combo.addItem(name, level)
        self.level_combo.currentIndexChanged.connect(self.show_records)
        self.filter = QtWidgets.QLineEdit(
            placeholderText="Filter",
            clearButtonEnabled=True,
            toolTip="Shows the records that contain the text, in any case",
        )
        refilter = QtCore.QTimer(
            self,
            singleShot=True,
            interval=_REFILTER_DELAY // timedelta(milliseconds=1),
            timeout=self.show_records,
        )
        self.filter.textChanged.connect(refilter.start)
        clear = QtWidgets.QToolButton(
            text="Clear", autoRaise=True, toolTip="Remove the records"
        )
        clear.clicked.connect(self.clear)
        self.text = QtWidgets.QPlainTextEdit(readOnly=True)
        self.text.setMaximumBlockCount(_RECORDS)
        self.text.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        self.text.viewport().installEventFilter(self)
        # Follows the newest records while scrolled to them
        self.following = True
        bar = self.text.verticalScrollBar()
        bar.valueChanged.connect(
            lambda value: setattr(self, "following", value == bar.maximum())
        )
        bar.rangeChanged.connect(
            lambda _, maximum: bar.setValue(maximum) if self.following else None
        )
        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.level_combo)
        header.addWidget(self.filter, 1)
        header.addWidget(clear)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.text, 1)
        self.setMinimumHeight(_MIN_HEIGHT)
        QtCore.QTimer(
            self,
            interval=_POLL_INTERVAL // timedelta(milliseconds=1),
            timeout=self.read_records,
        ).start()

    def read_records(self) -> None:
        if records := self.sink.get_new_records():
            self.records.extend(records)
            if self.isVisible():
                self.append(self.matching(records))
            else:
                self.stale = True

    def matching(self, records: Iterable[merian.LogRecord]) -> list[merian.LogRecord]:
        level = self.level_combo.currentData()
        text = self.filter.text().casefold()
        return [
            record
            for record in records
            if record.level >= level
            and (
                text in record.message.casefold()
                or text in record.logger_name.casefold()
            )
        ]

    def append(self, records: list[merian.LogRecord]) -> None:
        if not records:
            return
        cursor = QtGui.QTextCursor(self.text.document())
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.beginEditBlock()
        for record in records:
            if self.shown:
                cursor.insertBlock()
            self.shown.append(record)
            cursor.insertText(f"{record.time:%H:%M:%S.%f}"[:-3] + " ", self.dim_format)
            cursor.insertText(
                f"{_LEVEL_NAMES[record.level]:<{_LEVEL_WIDTH}} ",
                self.level_formats[record.level],
            )
            if record.logger_name:
                cursor.insertText(f"{record.logger_name} ", self.dim_format)
            # A line separator keeps one block per record
            cursor.insertText(
                _BLOCK_BREAKS.sub("\u2028", record.message), self.message_format
            )
        cursor.endEditBlock()

    def show_records(self) -> None:
        self.stale = False
        self.following = True
        self.text.clear()
        self.shown.clear()
        self.append(self.matching(self.records))

    def clear(self) -> None:
        self.records.clear()
        self.show_records()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self.stale:
            self.show_records()
        if self.following:
            bar = self.text.verticalScrollBar()
            bar.setValue(bar.maximum())

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() != QtCore.QEvent.Type.ToolTip:
            return False
        block = self.text.cursorForPosition(event.pos()).block()
        number = block.blockNumber()
        below = (
            event.pos().y()
            > self.text.blockBoundingGeometry(block)
            .translated(self.text.contentOffset())
            .bottom()
        )
        record = self.shown[number] if number < len(self.shown) and not below else None
        if record is None or record.source_file is None:
            QtWidgets.QToolTip.hideText()
            return True
        source = f"{PurePath(record.source_file).name}:{record.source_line}"
        if record.source_function:
            source += f" in {record.source_function}"
        QtWidgets.QToolTip.showText(event.globalPos(), source, self.text)
        return True
