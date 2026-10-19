"""Raw mouse input via Wayland's relative pointer and pointer constraints protocols."""

import ctypes
import functools
import os
from collections.abc import Callable
from ctypes import c_char_p, c_int, c_int32, c_uint32, c_void_p

from PySide6 import QtGui

_MARSHAL_FLAG_DESTROY = 1
_DISPLAY_GET_REGISTRY = 1
_REGISTRY_BIND = 0
_SEAT_GET_POINTER = 0
_POINTER_RELEASE = 1
_DESTROY = 0
_GET_RELATIVE_POINTER = 1
_LOCK_POINTER = 1
_LIFETIME_PERSISTENT = 2

_POINTER_CONSTRAINTS = "zwp_pointer_constraints_v1"
_POINTER_WARP = "wp_pointer_warp_v1"
_RELATIVE_POINTER_MANAGER = "zwp_relative_pointer_manager_v1"

# QtWaylandClient::QWaylandNativeInterface::nativeResourceForWindow(QByteArray const&, QWindow*)
_NATIVE_RESOURCE_FOR_WINDOW = (
    "_ZN15QtWaylandClient23QWaylandNativeInterface23"
    "nativeResourceForWindowERK10QByteArrayP7QWindow"
)


class _Message(ctypes.Structure):
    _fields_ = [
        ("name", c_char_p),
        ("signature", c_char_p),
        ("types", ctypes.POINTER(c_void_p)),
    ]


class _Interface(ctypes.Structure):
    _fields_ = [
        ("name", c_char_p),
        ("version", c_int),
        ("method_count", c_int),
        ("methods", ctypes.POINTER(_Message)),
        ("event_count", c_int),
        ("events", ctypes.POINTER(_Message)),
    ]


class _QByteArray(ctypes.Structure):
    """Qt 6's layout, raw data without an allocation."""

    _fields_ = [("d", c_void_p), ("ptr", c_char_p), ("size", ctypes.c_ssize_t)]


# The message arrays the interfaces point to
_DEFINITIONS = []


def _define(interface: _Interface, name: str, methods: list, events: list) -> None:
    """Methods and events are (name, signature, [interface or None per argument])."""

    def messages(entries: list) -> ctypes.Array:
        array = (_Message * max(len(entries), 1))()
        for message, (message_name, signature, types) in zip(array, entries):
            message.name, message.signature = message_name.encode(), signature.encode()
            message.types = (c_void_p * max(len(types), 1))(
                *[ctypes.addressof(entry) if entry else None for entry in types]
            )
        return array

    method_array, event_array = messages(methods), messages(events)
    _DEFINITIONS.append((method_array, event_array))
    interface.name, interface.version = name.encode(), 1
    interface.method_count, interface.methods = len(methods), method_array
    interface.event_count, interface.events = len(events), event_array


def _loaded_library(name: str) -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(name, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
    except OSError:
        return None


class _Protocols:
    def __init__(self, wl: ctypes.CDLL):
        self.wl = wl
        for name, restype in (
            ("wl_proxy_marshal_flags", c_void_p),
            ("wl_proxy_create_wrapper", c_void_p),
            ("wl_display_create_queue", c_void_p),
        ):
            getattr(wl, name).restype = restype
        wl.wl_proxy_get_version.restype = c_uint32
        wl.wl_proxy_get_version.argtypes = [c_void_p]
        core = {
            name: _Interface.in_dll(wl, f"wl_{name}_interface")
            for name in ("registry", "pointer", "surface", "region")
        }
        self.registry, self.pointer = core["registry"], core["pointer"]
        self.relative_pointer, self.relative_pointer_manager = (
            _Interface(),
            _Interface(),
        )
        self.locked_pointer, self.pointer_constraints = _Interface(), _Interface()
        _define(
            self.relative_pointer,
            "zwp_relative_pointer_v1",
            [("destroy", "", [])],
            [("relative_motion", "uuffff", [None] * 6)],
        )
        _define(
            self.relative_pointer_manager,
            _RELATIVE_POINTER_MANAGER,
            [
                ("destroy", "", []),
                (
                    "get_relative_pointer",
                    "no",
                    [self.relative_pointer, core["pointer"]],
                ),
            ],
            [],
        )
        _define(
            self.locked_pointer,
            "zwp_locked_pointer_v1",
            [
                ("destroy", "", []),
                ("set_cursor_position_hint", "ff", [None, None]),
                ("set_region", "?o", [core["region"]]),
            ],
            [("locked", "", []), ("unlocked", "", [])],
        )
        _define(
            self.pointer_constraints,
            _POINTER_CONSTRAINTS,
            [
                ("destroy", "", []),
                (
                    "lock_pointer",
                    "noo?ou",
                    [
                        self.locked_pointer,
                        core["surface"],
                        core["pointer"],
                        core["region"],
                        None,
                    ],
                ),
            ],
            [],
        )

    def marshal(
        self,
        proxy: int,
        opcode: int,
        interface: _Interface | None,
        *arguments,
        version: int | None = None,
        flags: int = 0,
    ) -> int:
        if version is None:
            version = self.wl.wl_proxy_get_version(c_void_p(proxy))
        return self.wl.wl_proxy_marshal_flags(
            c_void_p(proxy),
            c_uint32(opcode),
            ctypes.byref(interface) if interface else None,
            c_uint32(version),
            c_uint32(flags),
            *arguments,
        )

    def destroy(self, proxy: int, opcode: int = _DESTROY) -> None:
        self.marshal(proxy, opcode, None, flags=_MARSHAL_FLAG_DESTROY)


def _surface_lookup() -> Callable[[QtGui.QWindow], int] | None:
    gui, client = (
        _loaded_library("libQt6Gui.so.6"),
        _loaded_library("libQt6WaylandClient.so.6"),
    )
    try:
        import shiboken6

        native_interface = gui._ZN15QGuiApplication23platformNativeInterfaceEv
        resource = getattr(client, _NATIVE_RESOURCE_FOR_WINDOW)
    except (AttributeError, ImportError):
        return None
    native_interface.restype = c_void_p
    resource.restype = c_void_p
    resource.argtypes = [c_void_p, ctypes.POINTER(_QByteArray), c_void_p]
    name = b"surface"

    def lookup(window: QtGui.QWindow) -> int:
        return (
            resource(
                native_interface(),
                ctypes.byref(_QByteArray(None, name, len(name))),
                shiboken6.getCppPointer(window)[0],
            )
            or 0
        )

    return lookup


def _qt_warps() -> bool:
    client = _loaded_library("libQt6WaylandClient.so.6")
    return client is not None and hasattr(
        client, "_ZN9QtWayland18wp_pointer_warp_v19interfaceEv"
    )


class _WaylandPointer:
    def __init__(self, protocols: _Protocols, display: int, seat: int):
        self.protocols, self.display, self.seat = protocols, display, seat
        wl = protocols.wl
        # Globals are listed on their own queue, Qt's would dispatch Qt's events
        queue = wl.wl_display_create_queue(c_void_p(display))
        display_wrapper = wl.wl_proxy_create_wrapper(c_void_p(display))
        wl.wl_proxy_set_queue(c_void_p(display_wrapper), c_void_p(queue))
        self.advertised: dict[str, int] = {}
        registry = protocols.marshal(
            display_wrapper, _DISPLAY_GET_REGISTRY, protocols.registry, None
        )
        on_global = ctypes.CFUNCTYPE(
            None, c_void_p, c_void_p, c_uint32, c_char_p, c_uint32
        )(
            lambda _data, _proxy, name, interface, _version: (
                self.advertised.__setitem__(interface.decode(), name)
            )
        )
        on_global_remove = ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_uint32)(
            lambda _data, _proxy, _name: None
        )
        listener = self.listen(registry, on_global, on_global_remove)
        wl.wl_display_roundtrip_queue(c_void_p(display), c_void_p(queue))
        self.manager = self.bind(registry, protocols.relative_pointer_manager)
        self.constraints = self.bind(registry, protocols.pointer_constraints)
        # Objects they create join Qt's queue, Qt's event loop delivers their events
        for proxy in (self.manager, self.constraints):
            if proxy:
                wl.wl_proxy_set_queue(c_void_p(proxy), None)
        wl.wl_proxy_destroy(c_void_p(registry))
        wl.wl_proxy_wrapper_destroy(c_void_p(display_wrapper))
        wl.wl_event_queue_destroy(c_void_p(queue))
        del listener
        self.surface_of = _surface_lookup() if self.constraints else None
        self.warps = _POINTER_WARP in self.advertised and _qt_warps()
        self.motion: Callable[[float, float], None] | None = None
        self.motion_listener = None
        self.started: tuple[int, int, int | None] | None = None

    @property
    def relative(self) -> bool:
        return self.manager is not None

    @property
    def locks(self) -> bool:
        return self.surface_of is not None

    def bind(self, registry: int, interface: _Interface) -> int | None:
        if (name := self.advertised.get(interface.name.decode())) is None:
            return None
        return self.protocols.marshal(
            registry,
            _REGISTRY_BIND,
            interface,
            c_uint32(name),
            c_char_p(interface.name),
            c_uint32(1),
            None,
            version=1,
        )

    def listen(self, proxy: int, *callbacks) -> tuple:
        listener = (c_void_p * len(callbacks))(
            *[ctypes.cast(callback, c_void_p) for callback in callbacks]
        )
        self.protocols.wl.wl_proxy_add_listener(c_void_p(proxy), listener, None)
        return listener, callbacks

    def start(
        self, window: QtGui.QWindow, motion: Callable[[float, float], None]
    ) -> None:
        """
        Calls motion(dx, dy), unaccelerated in logical pixels, from Qt's event loop
        until stop().
        """
        self.stop()
        protocols = self.protocols
        pointer = protocols.marshal(
            self.seat, _SEAT_GET_POINTER, protocols.pointer, None
        )
        relative = protocols.marshal(
            self.manager,
            _GET_RELATIVE_POINTER,
            protocols.relative_pointer,
            None,
            c_void_p(pointer),
        )
        on_motion = ctypes.CFUNCTYPE(
            None,
            c_void_p,
            c_void_p,
            c_uint32,
            c_uint32,
            c_int32,
            c_int32,
            c_int32,
            c_int32,
        )(
            lambda _data, _proxy, _time_high, _time_low, _dx, _dy, raw_dx, raw_dy: (
                self.motion and self.motion(raw_dx / 256, raw_dy / 256)
            )
        )
        self.motion_listener = self.listen(relative, on_motion)
        locked = None
        if self.locks and (surface := self.surface_of(window)):
            locked = protocols.marshal(
                self.constraints,
                _LOCK_POINTER,
                protocols.locked_pointer,
                None,
                c_void_p(surface),
                c_void_p(pointer),
                None,
                c_uint32(_LIFETIME_PERSISTENT),
            )
        self.motion, self.started = motion, (pointer, relative, locked)
        protocols.wl.wl_display_flush(c_void_p(self.display))

    def stop(self) -> None:
        if self.started is None:
            return
        pointer, relative, locked = self.started
        if locked:
            self.protocols.destroy(locked)
        self.protocols.destroy(relative)
        self.protocols.destroy(pointer, _POINTER_RELEASE)
        self.protocols.wl.wl_display_flush(c_void_p(self.display))
        self.motion = self.started = self.motion_listener = None


@functools.cache
def wayland_pointer() -> _WaylandPointer | None:
    """Returns None off Wayland or with a libwayland-client older than 1.20."""
    if not QtGui.QGuiApplication.platformName().startswith("wayland"):
        return None
    wl = _loaded_library("libwayland-client.so.0")
    if wl is None:
        return None
    try:
        native = QtGui.QGuiApplication.instance().nativeInterface()
        if (
            not isinstance(native, QtGui.QNativeInterface.QWaylandApplication)
            or not native.seat()
        ):
            return None
        return _WaylandPointer(_Protocols(wl), native.display(), native.seat())
    except (AttributeError, ValueError):
        return None
