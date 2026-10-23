from PySide6 import QtCore, QtWidgets

import merian

_Modifier = QtCore.Qt.KeyboardModifier
SHORTCUT_MODIFIERS = {
    "Alt": _Modifier.AltModifier,
    "Ctrl": _Modifier.ControlModifier,
    "Shift+Alt": _Modifier.ShiftModifier | _Modifier.AltModifier,
    "Ctrl+Alt": _Modifier.ControlModifier | _Modifier.AltModifier,
}
_DEFAULT_LOG_LEVEL = "info"
_DEFAULT_SHORTCUT_MODIFIERS = "Shift+Alt"
_LOG_LEVEL_KEY = "log level"
_SHORTCUT_MODIFIERS_KEY = "shortcut modifiers"


def _row(label: str, field: QtWidgets.QWidget, tip: str) -> QtWidgets.QWidget:
    row = QtWidgets.QWidget(objectName="propertyRow", toolTip=tip)
    row.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(6, 1, 2, 1)
    layout.setSpacing(10)
    layout.addWidget(QtWidgets.QLabel(label, objectName="propertyLabel"), 2)
    layout.addWidget(field, 3)
    return row


def _select(combo: QtWidgets.QComboBox, text: str, default: str) -> None:
    index = combo.findText(text)
    combo.setCurrentIndex(index if index >= 0 else combo.findText(default))


class PreferencesView(QtWidgets.QWidget):
    shortcut_modifiers_changed = QtCore.Signal(object)

    def __init__(self, settings: QtCore.QSettings):
        super().__init__()
        self.settings = settings
        self.modifiers_combo = QtWidgets.QComboBox()
        self.modifiers_combo.addItems(list(SHORTCUT_MODIFIERS))
        _select(
            self.modifiers_combo,
            str(settings.value(_SHORTCUT_MODIFIERS_KEY, "")),
            _DEFAULT_SHORTCUT_MODIFIERS,
        )
        self.modifiers_combo.currentTextChanged.connect(self.store_shortcut_modifiers)
        self.log_level_combo = QtWidgets.QComboBox()
        self.log_level_combo.addItems([level.name.lower() for level in merian.LogLevel])
        _select(
            self.log_level_combo,
            str(settings.value(_LOG_LEVEL_KEY, "")),
            _DEFAULT_LOG_LEVEL,
        )
        self.log_level_combo.currentTextChanged.connect(self.store_log_level)
        merian.set_log_level(self.log_level())

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            _row(
                "shortcut modifier",
                self.modifiers_combo,
                "Held with Space, Right and Backspace to play and pause, step and release "
                "the mouse, also in a graph's view.",
            )
        )
        layout.addWidget(
            _row(
                "log level",
                self.log_level_combo,
                "The least level merian logs, to the console and the log",
            )
        )
        layout.addStretch()

    def shortcut_modifiers(self) -> QtCore.Qt.KeyboardModifier:
        return SHORTCUT_MODIFIERS[self.modifiers_combo.currentText()]

    def log_level(self) -> merian.LogLevel:
        return merian.LogLevel[self.log_level_combo.currentText().upper()]

    def store_shortcut_modifiers(self, name: str) -> None:
        self.settings.setValue(_SHORTCUT_MODIFIERS_KEY, name)
        self.shortcut_modifiers_changed.emit(self.shortcut_modifiers())

    def store_log_level(self, name: str) -> None:
        self.settings.setValue(_LOG_LEVEL_KEY, name)
        merian.set_log_level(self.log_level())
