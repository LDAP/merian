"""Raw mouse input via Win32 Raw Input, with the cursor hidden and clipped in place."""

import ctypes
import functools
from collections.abc import Callable
from ctypes import wintypes

from PySide6 import QtCore, QtGui

_WM_INPUT = 0x00FF
_RID_INPUT = 0x10000003
_RIM_TYPEMOUSE = 0
_RIDEV_REMOVE = 0x00000001
_MOUSE_MOVE_ABSOLUTE = 0x01
_MOUSE_VIRTUAL_DESKTOP = 0x02
_GA_ROOT = 2
_SM_CXSCREEN, _SM_CYSCREEN = 0, 1
_SM_XVIRTUALSCREEN, _SM_YVIRTUALSCREEN = 76, 77
_SM_CXVIRTUALSCREEN, _SM_CYVIRTUALSCREEN = 78, 79
_GENERIC_DESKTOP = 0x01
_GENERIC_MOUSE = 0x02
# Normalized coordinates of absolute motion
_ABSOLUTE_RANGE = 65535


class _RawInputDevice(ctypes.Structure):
    _fields_ = [
        ("usage_page", wintypes.USHORT),
        ("usage", wintypes.USHORT),
        ("flags", wintypes.DWORD),
        ("target", wintypes.HWND),
    ]


class _RawInputHeader(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("size", wintypes.DWORD),
        ("device", wintypes.HANDLE),
        ("wparam", wintypes.WPARAM),
    ]


class _RawMouse(ctypes.Structure):
    _fields_ = [
        ("flags", wintypes.USHORT),
        ("buttons", wintypes.ULONG),
        ("raw_buttons", wintypes.ULONG),
        ("last_x", wintypes.LONG),
        ("last_y", wintypes.LONG),
        ("extra_information", wintypes.ULONG),
    ]


class _RawInput(ctypes.Structure):
    _fields_ = [("header", _RawInputHeader), ("mouse", _RawMouse)]


def _user32() -> ctypes.WinDLL:
    user32 = ctypes.WinDLL("user32")
    user32.RegisterRawInputDevices.argtypes = [
        ctypes.POINTER(_RawInputDevice),
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.RegisterRawInputDevices.restype = wintypes.BOOL
    user32.GetRawInputData.argtypes = [
        wintypes.HANDLE,
        wintypes.UINT,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.UINT),
        wintypes.UINT,
    ]
    user32.GetRawInputData.restype = wintypes.UINT
    user32.ClipCursor.argtypes = [ctypes.POINTER(wintypes.RECT)]
    user32.GetClipCursor.argtypes = [ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user32.GetAncestor.restype = wintypes.HWND
    return user32


class _WindowsPointer(QtCore.QAbstractNativeEventFilter):
    def __init__(self, user32: ctypes.WinDLL):
        super().__init__()
        self.user32 = user32
        self.raw_input = _RawInput()
        self.window: QtGui.QWindow | None = None
        self.hwnd = 0
        self.motion: Callable[[float, float], None] | None = None
        self.displaced: Callable[[], None] | None = None
        self.absolute: tuple[int, int] | None = None
        self.restore = wintypes.POINT()
        self.clip: tuple[int, int] | None = None

    def start(
        self,
        window: QtGui.QWindow,
        motion: Callable[[float, float], None],
        displaced: Callable[[], None],
    ) -> bool:
        """
        Calls motion(dx, dy), unaccelerated in logical pixels, from Qt's event loop
        until stop(motion), or displaced() once another start() takes over. The cursor
        stays hidden, where hold() puts it. Returns False without raw input.
        """
        previous = self.displaced if self.motion not in (None, motion) else None
        self.stop(self.motion)
        if previous:
            previous()
        hwnd = self.user32.GetAncestor(int(window.winId()), _GA_ROOT)
        if not hwnd:
            return False
        device = _RawInputDevice(_GENERIC_DESKTOP, _GENERIC_MOUSE, 0, hwnd)
        if not self.user32.RegisterRawInputDevices(
            ctypes.byref(device), 1, ctypes.sizeof(device)
        ):
            return False
        self.window, self.hwnd, self.absolute = window, hwnd, None
        self.motion, self.displaced = motion, displaced
        self.user32.GetCursorPos(ctypes.byref(self.restore))
        QtCore.QCoreApplication.instance().installNativeEventFilter(self)
        QtGui.QGuiApplication.setOverrideCursor(QtCore.Qt.CursorShape.BlankCursor)
        return True

    def hold(self, position: QtCore.QPointF) -> None:
        """Clips the cursor to position in the window, released while in background."""
        if self.window is None:
            return
        if self.user32.GetForegroundWindow() != self.hwnd:
            self.unclip()
            return
        ratio = self.window.devicePixelRatio()
        point = wintypes.POINT(round(position.x() * ratio), round(position.y() * ratio))
        self.user32.ClientToScreen(self.hwnd, ctypes.byref(point))
        current = wintypes.RECT()
        self.user32.GetClipCursor(ctypes.byref(current))
        if self.clip == (point.x, point.y) and (current.left, current.top) == self.clip:
            return
        rect = wintypes.RECT(point.x, point.y, point.x + 1, point.y + 1)
        if self.user32.ClipCursor(ctypes.byref(rect)):
            self.clip = (point.x, point.y)

    def unclip(self) -> None:
        if self.clip is not None:
            self.user32.ClipCursor(None)
            self.clip = None

    def stop(self, motion: Callable[[float, float], None]) -> None:
        if self.window is None or motion != self.motion:
            return
        self.unclip()
        if self.user32.GetForegroundWindow() == self.hwnd:
            self.user32.SetCursorPos(self.restore.x, self.restore.y)
        QtGui.QGuiApplication.restoreOverrideCursor()
        QtCore.QCoreApplication.instance().removeNativeEventFilter(self)
        device = _RawInputDevice(_GENERIC_DESKTOP, _GENERIC_MOUSE, _RIDEV_REMOVE, None)
        self.user32.RegisterRawInputDevices(
            ctypes.byref(device), 1, ctypes.sizeof(device)
        )
        self.window = self.motion = self.displaced = None

    def nativeEventFilter(self, event_type, message) -> tuple[bool, int]:
        # also passed once from the event dispatcher
        if bytes(event_type) != b"windows_generic_MSG":
            return False, 0
        msg = wintypes.MSG.from_address(int(message))
        if msg.message != _WM_INPUT or msg.hWnd != self.hwnd or self.motion is None:
            return False, 0
        size = wintypes.UINT(ctypes.sizeof(self.raw_input))
        read = self.user32.GetRawInputData(
            msg.lParam,
            _RID_INPUT,
            ctypes.byref(self.raw_input),
            ctypes.byref(size),
            ctypes.sizeof(_RawInputHeader),
        )
        if read != wintypes.UINT(-1).value and self.raw_input.header.type == (
            _RIM_TYPEMOUSE
        ):
            self.move(self.raw_input.mouse)
        return False, 0

    def move(self, mouse: _RawMouse) -> None:
        if mouse.flags & _MOUSE_MOVE_ABSOLUTE:
            metrics = self.user32.GetSystemMetrics
            if mouse.flags & _MOUSE_VIRTUAL_DESKTOP:
                left, top = metrics(_SM_XVIRTUALSCREEN), metrics(_SM_YVIRTUALSCREEN)
                width = metrics(_SM_CXVIRTUALSCREEN)
                height = metrics(_SM_CYVIRTUALSCREEN)
            else:
                left, top = 0, 0
                width, height = metrics(_SM_CXSCREEN), metrics(_SM_CYSCREEN)
            position = (
                left + mouse.last_x * width // _ABSOLUTE_RANGE,
                top + mouse.last_y * height // _ABSOLUTE_RANGE,
            )
            last, self.absolute = self.absolute, position
            if last is None:
                return
            dx, dy = position[0] - last[0], position[1] - last[1]
        else:
            dx, dy = mouse.last_x, mouse.last_y
        if dx or dy:
            ratio = self.window.devicePixelRatio()
            self.motion(dx / ratio, dy / ratio)


@functools.cache
def windows_pointer() -> _WindowsPointer | None:
    if QtGui.QGuiApplication.platformName() != "windows":
        return None
    return _WindowsPointer(_user32())
