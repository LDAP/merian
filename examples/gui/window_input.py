import enum
import functools
import operator
from collections.abc import Callable

from PySide6 import QtCore, QtGui, QtWidgets

import merian.graph as mg

from .wayland_pointer import wayland_pointer

_Qt = QtCore.Qt
_K = mg.Key
_KEYS = {
    _Qt.Key.Key_Space: _K.SPACE,
    _Qt.Key.Key_Apostrophe: _K.APOSTROPHE,
    _Qt.Key.Key_Comma: _K.COMMA,
    _Qt.Key.Key_Minus: _K.MINUS,
    _Qt.Key.Key_Period: _K.PERIOD,
    _Qt.Key.Key_Slash: _K.SLASH,
    _Qt.Key.Key_Semicolon: _K.SEMICOLON,
    _Qt.Key.Key_Equal: _K.EQUAL,
    _Qt.Key.Key_BracketLeft: _K.LEFT_BRACKET,
    _Qt.Key.Key_Backslash: _K.BACKSLASH,
    _Qt.Key.Key_BracketRight: _K.RIGHT_BRACKET,
    _Qt.Key.Key_QuoteLeft: _K.GRAVE_ACCENT,
    _Qt.Key.Key_Escape: _K.ESCAPE,
    _Qt.Key.Key_Return: _K.ENTER,
    _Qt.Key.Key_Enter: _K.ENTER,
    _Qt.Key.Key_Tab: _K.TAB,
    _Qt.Key.Key_Backtab: _K.TAB,
    _Qt.Key.Key_Backspace: _K.BACKSPACE,
    _Qt.Key.Key_Insert: _K.INSERT,
    _Qt.Key.Key_Delete: _K.DELETE_KEY,
    _Qt.Key.Key_Right: _K.RIGHT,
    _Qt.Key.Key_Left: _K.LEFT,
    _Qt.Key.Key_Down: _K.DOWN,
    _Qt.Key.Key_Up: _K.UP,
    _Qt.Key.Key_PageUp: _K.PAGE_UP,
    _Qt.Key.Key_PageDown: _K.PAGE_DOWN,
    _Qt.Key.Key_Home: _K.HOME,
    _Qt.Key.Key_End: _K.END,
    _Qt.Key.Key_CapsLock: _K.CAPS_LOCK,
    _Qt.Key.Key_ScrollLock: _K.SCROLL_LOCK,
    _Qt.Key.Key_NumLock: _K.NUM_LOCK,
    _Qt.Key.Key_Print: _K.PRINT_SCREEN,
    _Qt.Key.Key_Pause: _K.PAUSE,
    _Qt.Key.Key_Shift: _K.LEFT_SHIFT,
    _Qt.Key.Key_Control: _K.LEFT_CTRL,
    _Qt.Key.Key_Alt: _K.LEFT_ALT,
    _Qt.Key.Key_Meta: _K.LEFT_SUPER,
    _Qt.Key.Key_AltGr: _K.RIGHT_ALT,
    _Qt.Key.Key_Menu: _K.MENU,
}
# The shifted symbols of a US layout
_KEYS |= {
    _Qt.Key.Key_Exclam: _K.NUM_1,
    _Qt.Key.Key_At: _K.NUM_2,
    _Qt.Key.Key_NumberSign: _K.NUM_3,
    _Qt.Key.Key_Dollar: _K.NUM_4,
    _Qt.Key.Key_Percent: _K.NUM_5,
    _Qt.Key.Key_AsciiCircum: _K.NUM_6,
    _Qt.Key.Key_Ampersand: _K.NUM_7,
    _Qt.Key.Key_Asterisk: _K.NUM_8,
    _Qt.Key.Key_ParenLeft: _K.NUM_9,
    _Qt.Key.Key_ParenRight: _K.NUM_0,
    _Qt.Key.Key_Underscore: _K.MINUS,
    _Qt.Key.Key_Plus: _K.EQUAL,
    _Qt.Key.Key_BraceLeft: _K.LEFT_BRACKET,
    _Qt.Key.Key_BraceRight: _K.RIGHT_BRACKET,
    _Qt.Key.Key_Bar: _K.BACKSLASH,
    _Qt.Key.Key_Colon: _K.SEMICOLON,
    _Qt.Key.Key_QuoteDbl: _K.APOSTROPHE,
    _Qt.Key.Key_Less: _K.COMMA,
    _Qt.Key.Key_Greater: _K.PERIOD,
    _Qt.Key.Key_Question: _K.SLASH,
    _Qt.Key.Key_AsciiTilde: _K.GRAVE_ACCENT,
}
_KEYS |= {_Qt.Key(_Qt.Key.Key_A.value + i): _K(_K.A.value + i) for i in range(26)}
_KEYS |= {_Qt.Key(_Qt.Key.Key_0.value + i): _K(_K.NUM_0.value + i) for i in range(10)}
_KEYS |= {_Qt.Key(_Qt.Key.Key_F1.value + i): _K(_K.F1.value + i) for i in range(12)}
_KEYPAD_KEYS = {
    _Qt.Key(_Qt.Key.Key_0.value + i): _K(_K.KP_0.value + i) for i in range(10)
} | {
    _Qt.Key.Key_Period: _K.KP_DECIMAL,
    _Qt.Key.Key_Comma: _K.KP_DECIMAL,
    _Qt.Key.Key_Slash: _K.KP_DIVIDE,
    _Qt.Key.Key_Asterisk: _K.KP_MULTIPLY,
    _Qt.Key.Key_Minus: _K.KP_SUBTRACT,
    _Qt.Key.Key_Plus: _K.KP_ADD,
    _Qt.Key.Key_Enter: _K.KP_ENTER,
    _Qt.Key.Key_Equal: _K.KP_EQUAL,
}
# By the integer QKeyEvent.key() returns
_KEYS = {qt_key.value: key for qt_key, key in _KEYS.items()}
_KEYPAD_KEYS = {qt_key.value: key for qt_key, key in _KEYPAD_KEYS.items()}
_MODIFIERS = {
    _Qt.KeyboardModifier.ShiftModifier: mg.ModKey.SHIFT,
    _Qt.KeyboardModifier.ControlModifier: mg.ModKey.CONTROL,
    _Qt.KeyboardModifier.AltModifier: mg.ModKey.ALT,
    _Qt.KeyboardModifier.MetaModifier: mg.ModKey.SUPER,
}
_MODIFIER_MASK = functools.reduce(operator.or_, _MODIFIERS)

_BUTTONS = {
    _Qt.MouseButton.LeftButton: mg.MouseButton.MOUSE1,
    _Qt.MouseButton.RightButton: mg.MouseButton.MOUSE2,
    _Qt.MouseButton.MiddleButton: mg.MouseButton.MOUSE3,
    _Qt.MouseButton.BackButton: mg.MouseButton.MOUSE4,
    _Qt.MouseButton.ForwardButton: mg.MouseButton.MOUSE5,
}
_CURSORS = {
    mg.CursorShape.ARROW: _Qt.CursorShape.ArrowCursor,
    mg.CursorShape.TEXT_INPUT: _Qt.CursorShape.IBeamCursor,
    mg.CursorShape.RESIZE_ALL: _Qt.CursorShape.SizeAllCursor,
    mg.CursorShape.RESIZE_NS: _Qt.CursorShape.SizeVerCursor,
    mg.CursorShape.RESIZE_EW: _Qt.CursorShape.SizeHorCursor,
    mg.CursorShape.RESIZE_NESW: _Qt.CursorShape.SizeBDiagCursor,
    mg.CursorShape.RESIZE_NWSE: _Qt.CursorShape.SizeFDiagCursor,
    mg.CursorShape.HAND: _Qt.CursorShape.PointingHandCursor,
    mg.CursorShape.NOT_ALLOWED: _Qt.CursorShape.ForbiddenCursor,
    mg.CursorShape.HIDDEN: _Qt.CursorShape.BlankCursor,
}
_RECENTER_DISTANCE = 100


def _key_of(event: QtGui.QKeyEvent) -> mg.Key:
    if (
        event.modifiers() & _Qt.KeyboardModifier.KeypadModifier
        and event.key() in _KEYPAD_KEYS
    ):
        return _KEYPAD_KEYS[event.key()]
    return _KEYS.get(event.key(), _K.UNKNOWN)


def _mods_of(modifiers: _Qt.KeyboardModifier) -> int:
    return sum(int(mod.value) for flag, mod in _MODIFIERS.items() if modifiers & flag)


class _RawMotion(enum.Enum):
    RELATIVE = enum.auto()
    WARPED = enum.auto()
    ABSOLUTE = enum.auto()


def _raw_motion() -> tuple[_RawMotion, bool, str | None]:
    if not QtGui.QGuiApplication.platformName().startswith("wayland"):
        return _RawMotion.WARPED, True, None
    pointer = wayland_pointer()
    if pointer and pointer.relative:
        if pointer.locks or pointer.warps:
            return _RawMotion.RELATIVE, not pointer.locks, None
        return (
            _RawMotion.RELATIVE,
            False,
            "The compositor cannot hold the pointer: "
            "raw mouse input pauses while the cursor is outside the window",
        )
    if pointer and pointer.warps:
        return _RawMotion.WARPED, True, None
    return (
        _RawMotion.ABSOLUTE,
        False,
        "The compositor reports no relative pointer motion: "
        "raw mouse input stops at the screen's edges",
    )


class _WarpedCursor:
    """
    Motion from the positions of a cursor that is warped back whenever it strays. Events
    from before a warp may still arrive after it: positions at least as near the warp's
    target as its origin count from the target, the others from the last position.
    """

    def __init__(self, position: QtCore.QPointF):
        self.last = position
        # (target, origin) until a position from after the warp arrives
        self.warp: tuple[QtCore.QPointF, QtCore.QPointF] | None = None

    def warp_to(self, target: QtCore.QPointF) -> None:
        self.warp = (target, self.last)

    def move(self, position: QtCore.QPointF) -> QtCore.QPointF:
        """Returns the motion since the last position."""
        last = self.last
        if self.warp is not None:
            target, origin = self.warp
            if (position - target).manhattanLength() <= (
                position - origin
            ).manhattanLength():
                last, self.warp = target, None
        self.last = position
        return position - last


class WindowInput(QtCore.QObject):
    """
    Sends the input of a widget to a window: keys while the widget has the focus, the
    mouse over area(), the part of mouse_source that shows the window, but the key
    combinations in shortcuts (QKeyCombination.toCombined()), which trigger shortcuts.
    While the window asks for raw input it does not get, a click captures the mouse.
    """

    message = QtCore.Signal(str)

    def __init__(
        self,
        widget: QtWidgets.QWidget,
        area: Callable[[], QtCore.QRectF],
        mouse_source: QtWidgets.QWidget | None = None,
    ):
        super().__init__(widget)
        self.widget = widget
        self.mouse_source = mouse_source or widget
        self.area = area
        self.window: mg.HostWindow | None = None
        self.shortcuts: set[int] = set()
        self.raw = False
        self.released = False
        self.pressed_keys: dict[int, tuple[mg.Key, int]] = {}
        self.pressed_buttons: set[mg.MouseButton] = set()
        self.position = (0.0, 0.0)
        self.cursor = QtCore.QPointF()
        self.raw_start = QtCore.QPoint()
        self.motion = _RawMotion.ABSOLUTE
        self.recenters = False
        self.raw_cursor = _WarpedCursor(self.cursor)
        self.warned = False
        self.focus_policy = widget.focusPolicy()
        self.mouse_tracking = self.mouse_source.hasMouseTracking()
        widget.installEventFilter(self)
        if self.mouse_source is not widget:
            self.mouse_source.installEventFilter(self)

    def set_window(self, window: mg.HostWindow | None) -> None:
        self.release_all()
        self.set_raw(False)
        self.released = False
        self.window = window
        self.widget.setFocusPolicy(
            _Qt.FocusPolicy.StrongFocus if window else self.focus_policy
        )
        self.mouse_source.setMouseTracking(window is not None or self.mouse_tracking)
        self.mouse_source.unsetCursor()

    def update(self) -> None:
        """Follows the cursor and raw input the window asks for."""
        if self.window is None:
            return
        grabbed = self.window.mouse_grabbed
        self.released &= grabbed
        raw = grabbed and self.widget.hasFocus() and not self.released
        self.set_raw(raw)
        if grabbed and not raw:
            self.mouse_source.setCursor(_Qt.CursorShape.ArrowCursor)
        elif not raw:
            self.mouse_source.setCursor(
                _CURSORS.get(self.window.cursor, _Qt.CursorShape.ArrowCursor)
            )

    def release_mouse(self) -> bool:
        """Gives up raw input until the next click; returns whether it had it."""
        if not self.raw:
            return False
        self.released = True
        self.update()
        return True

    def held_back(self) -> bool:
        """
        Whether the window asks for raw input it does not get; it takes cursor positions
        for motion then, so the mouse is not sent.
        """
        return self.window.mouse_grabbed and not self.raw

    def set_raw(self, raw: bool) -> None:
        if raw == self.raw:
            return
        if raw:
            self.motion, self.recenters, missing = _raw_motion()
            self.raw = True
            if missing and not self.warned:
                self.warned = True
                self.message.emit(missing)
            self.mouse_source.setCursor(_Qt.CursorShape.BlankCursor)
            self.mouse_source.grabMouse()
            self.raw_start = QtGui.QCursor.pos()
            self.cursor = QtCore.QPointF(
                self.mouse_source.mapFromGlobal(self.raw_start)
            )
            self.raw_cursor = _WarpedCursor(self.cursor)
            if self.motion == _RawMotion.RELATIVE:
                wayland_pointer().start(
                    self.widget.window().windowHandle(), self.send_relative_motion
                )
            if self.recenters:
                self.recenter()
            return
        self.raw = False
        if self.window is not None and self.window.mouse_grabbed:
            self.release_buttons()
        if self.motion == _RawMotion.RELATIVE:
            wayland_pointer().stop()
        if QtWidgets.QWidget.mouseGrabber() is self.mouse_source:
            self.mouse_source.releaseMouse()
        if self.recenters:
            QtGui.QCursor.setPos(self.raw_start)

    def recenter(self) -> None:
        center = self.area().center().toPoint()
        self.raw_cursor.warp_to(QtCore.QPointF(center))
        QtGui.QCursor.setPos(self.mouse_source.mapToGlobal(center))

    def pixel(self, position: QtCore.QPointF) -> tuple[float, float]:
        area = self.area()
        width, height = self.window.extent
        return (
            (position.x() - area.left()) / max(area.width(), 1) * width,
            (position.y() - area.top()) / max(area.height(), 1) * height,
        )

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if self.window is None:
            return False
        kind = event.type()
        Type = QtCore.QEvent.Type
        if watched is self.widget:
            if kind == Type.ShortcutOverride:
                modifiers = event.modifiers() & _MODIFIER_MASK
                if event.key() | modifiers.value not in self.shortcuts:
                    event.accept()
                return True
            if kind in (Type.KeyPress, Type.KeyRelease):
                self.send_key(event)
                return True
            if kind in (Type.FocusIn, Type.FocusOut):
                if kind == Type.FocusOut:
                    self.release_all()
                    self.set_raw(False)
                self.window.send_focus(kind == Type.FocusIn)
        if watched is self.mouse_source:
            if kind in (
                Type.MouseButtonPress,
                Type.MouseButtonDblClick,
                Type.MouseButtonRelease,
            ):
                self.widget.setFocus(_Qt.FocusReason.MouseFocusReason)
                self.cursor = event.position()
                if not self.held_back():
                    self.send_button(event)
                elif kind != Type.MouseButtonRelease:
                    self.released = False
                    self.update()
                return True
            if kind == Type.MouseMove:
                self.send_motion(event.position())
                return True
            if kind == Type.Wheel:
                if self.held_back():
                    return True
                steps = event.angleDelta()
                self.window.send_scroll(steps.x() / 120, steps.y() / 120)
                return True
            if kind == Type.ContextMenu:
                return True
        return False

    def send_button(self, event: QtGui.QMouseEvent) -> None:
        """Sends a release only for a sent press."""
        button = _BUTTONS.get(event.button())
        if button is None:
            return
        if event.type() != QtCore.QEvent.Type.MouseButtonRelease:
            self.pressed_buttons.add(button)
            status = mg.KeyStatus.PRESS
        elif button in self.pressed_buttons:
            self.pressed_buttons.discard(button)
            status = mg.KeyStatus.RELEASE
        else:
            return
        self.window.send_mouse_button(button, status, _mods_of(event.modifiers()))

    def send_key(self, event: QtGui.QKeyEvent) -> None:
        """
        A release sends the key of its press, as the mods may change the symbol, and only
        for a sent press.
        """
        code = event.nativeScanCode() or event.key()
        mods = _mods_of(event.modifiers())
        if event.type() == QtCore.QEvent.Type.KeyRelease:
            if not event.isAutoRepeat() and code in self.pressed_keys:
                key, _ = self.pressed_keys.pop(code)
                self.window.send_key(key, mg.KeyStatus.RELEASE, mods)
            return
        key, _ = self.pressed_keys.setdefault(code, (_key_of(event), mods))
        status = mg.KeyStatus.REPEAT if event.isAutoRepeat() else mg.KeyStatus.PRESS
        self.window.send_key(key, status, mods)
        for character in event.text():
            if character.isprintable():
                self.window.send_char(ord(character))

    def release_all(self) -> None:
        if self.window is not None:
            for key, mods in self.pressed_keys.values():
                self.window.send_key(key, mg.KeyStatus.RELEASE, mods)
        self.pressed_keys.clear()
        self.release_buttons()

    def release_buttons(self) -> None:
        if self.window is not None:
            for button in self.pressed_buttons:
                self.window.send_mouse_button(button, mg.KeyStatus.RELEASE, 0)
        self.pressed_buttons.clear()

    def send_motion(self, position: QtCore.QPointF) -> None:
        self.cursor = position
        if self.held_back():
            return
        if not self.raw:
            self.position = self.pixel(position)
            self.window.send_cursor(*self.position)
            return
        motion = self.raw_cursor.move(position)
        if self.motion != _RawMotion.RELATIVE and not motion.isNull():
            self.send_raw_motion(motion.x(), motion.y())
        if (
            self.recenters
            and self.raw_cursor.warp is None
            and (position - self.area().center()).manhattanLength() > _RECENTER_DISTANCE
        ):
            self.recenter()

    def send_relative_motion(self, dx: float, dy: float) -> None:
        if self.raw and self.window is not None:
            self.send_raw_motion(dx, dy)

    def send_raw_motion(self, dx: float, dy: float) -> None:
        """Sends a motion in logical pixels."""
        ratio = self.mouse_source.devicePixelRatioF()
        self.position = (self.position[0] + dx * ratio, self.position[1] + dy * ratio)
        self.window.send_cursor(*self.position)
