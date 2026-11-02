from PySide6 import QtCore, QtWidgets

import merian

_Modifier = QtCore.Qt.KeyboardModifier
SHORTCUT_MODIFIERS = {
    "Alt": _Modifier.AltModifier,
    "Ctrl": _Modifier.ControlModifier,
    "Shift+Alt": _Modifier.ShiftModifier | _Modifier.AltModifier,
    "Ctrl+Alt": _Modifier.ControlModifier | _Modifier.AltModifier,
}
SHADER_OPTIMIZATION_LEVELS = ["none", "default", "high", "maximal"]
SHADER_DEBUG_INFO = ["off", "on"]
_MERIAN = "merian"
_DEFAULT_LOG_LEVEL = "info"
_DEFAULT_SHORTCUT_MODIFIERS = "Shift+Alt"
_LOG_LEVEL_KEY = "log level"
_SHORTCUT_MODIFIERS_KEY = "shortcut modifiers"
_SHADER_OPTIMIZATION_KEY = "shader optimization"
_SHADER_DEBUG_INFO_KEY = "shader debug info"


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


def _option_combo(names: list[str], default: int, stored: str) -> QtWidgets.QComboBox:
    combo = QtWidgets.QComboBox()
    combo.addItem(f"{_MERIAN} ({names[default]})", None)
    for value, name in enumerate(names):
        combo.addItem(name, value)
    index = combo.findText(stored)
    combo.setCurrentIndex(max(index, 0))
    return combo


def _option_name(combo: QtWidgets.QComboBox) -> str:
    return _MERIAN if combo.currentData() is None else combo.currentText()


class PreferencesView(QtWidgets.QWidget):
    shortcut_modifiers_changed = QtCore.Signal(object)
    shader_options_changed = QtCore.Signal(object)

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
        self.optimization_combo = _option_combo(
            SHADER_OPTIMIZATION_LEVELS,
            merian.Context.DEFAULT_SHADER_OPTIMIZATION_LEVEL,
            str(settings.value(_SHADER_OPTIMIZATION_KEY, "")),
        )
        self.optimization_combo.currentIndexChanged.connect(
            self.store_shader_optimization
        )
        self.debug_info_combo = _option_combo(
            SHADER_DEBUG_INFO,
            int(merian.Context.DEFAULT_SHADER_DEBUG_INFO),
            str(settings.value(_SHADER_DEBUG_INFO_KEY, "")),
        )
        self.debug_info_combo.currentIndexChanged.connect(self.store_shader_debug_info)

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
        layout.addWidget(
            _row(
                "shader optimization",
                self.optimization_combo,
                "How far the shaders are optimized, 'merian' as merian was built. "
                "Changing it reconnects the graph, which compiles them anew.",
            )
        )
        layout.addWidget(
            _row(
                "shader debug info",
                self.debug_info_combo,
                "Whether the shaders carry debug information, 'merian' as merian was "
                "built. Changing it reconnects the graph, which compiles them anew.",
            )
        )
        layout.addStretch()

    def shortcut_modifiers(self) -> QtCore.Qt.KeyboardModifier:
        return SHORTCUT_MODIFIERS[self.modifiers_combo.currentText()]

    def log_level(self) -> merian.LogLevel:
        return merian.LogLevel[self.log_level_combo.currentText().upper()]

    def shader_options(self) -> tuple[int | None, bool | None]:
        debug_info = self.debug_info_combo.currentData()
        return (
            self.optimization_combo.currentData(),
            None if debug_info is None else bool(debug_info),
        )

    def store_shortcut_modifiers(self, name: str) -> None:
        self.settings.setValue(_SHORTCUT_MODIFIERS_KEY, name)
        self.shortcut_modifiers_changed.emit(self.shortcut_modifiers())

    def store_log_level(self, name: str) -> None:
        self.settings.setValue(_LOG_LEVEL_KEY, name)
        merian.set_log_level(self.log_level())

    def store_shader_optimization(self) -> None:
        self.settings.setValue(
            _SHADER_OPTIMIZATION_KEY, _option_name(self.optimization_combo)
        )
        self.shader_options_changed.emit(self.shader_options())

    def store_shader_debug_info(self) -> None:
        self.settings.setValue(
            _SHADER_DEBUG_INFO_KEY, _option_name(self.debug_info_combo)
        )
        self.shader_options_changed.emit(self.shader_options())
