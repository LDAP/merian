// Python.h must precede the standard headers
#include "bindings.hpp"
#include "transfer.hpp"

#include "merian/vk/extension/extension_host_windows.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

namespace merian::python {

using namespace nb::literals;

namespace {

void bind_input_enums(nb::module_& m) {
    using Key = InputController::Key;
    nb::enum_<Key> key(m, "Key", "A key, independent of the keyboard layout.");
#define MERIAN_KEY(name) key.value(#name, Key::name)
    // clang-format off
    MERIAN_KEY(UNKNOWN);
    MERIAN_KEY(SPACE); MERIAN_KEY(APOSTROPHE); MERIAN_KEY(COMMA); MERIAN_KEY(MINUS);
    MERIAN_KEY(PERIOD); MERIAN_KEY(SLASH); MERIAN_KEY(SEMICOLON); MERIAN_KEY(EQUAL);
    MERIAN_KEY(LEFT_BRACKET); MERIAN_KEY(BACKSLASH); MERIAN_KEY(RIGHT_BRACKET);
    MERIAN_KEY(GRAVE_ACCENT);
    MERIAN_KEY(NUM_0); MERIAN_KEY(NUM_1); MERIAN_KEY(NUM_2); MERIAN_KEY(NUM_3); MERIAN_KEY(NUM_4);
    MERIAN_KEY(NUM_5); MERIAN_KEY(NUM_6); MERIAN_KEY(NUM_7); MERIAN_KEY(NUM_8); MERIAN_KEY(NUM_9);
    MERIAN_KEY(A); MERIAN_KEY(B); MERIAN_KEY(C); MERIAN_KEY(D); MERIAN_KEY(E); MERIAN_KEY(F);
    MERIAN_KEY(G); MERIAN_KEY(H); MERIAN_KEY(I); MERIAN_KEY(J); MERIAN_KEY(K); MERIAN_KEY(L);
    MERIAN_KEY(M); MERIAN_KEY(N); MERIAN_KEY(O); MERIAN_KEY(P); MERIAN_KEY(Q); MERIAN_KEY(R);
    MERIAN_KEY(S); MERIAN_KEY(T); MERIAN_KEY(U); MERIAN_KEY(V); MERIAN_KEY(W); MERIAN_KEY(X);
    MERIAN_KEY(Y); MERIAN_KEY(Z);
    MERIAN_KEY(ESCAPE); MERIAN_KEY(ENTER); MERIAN_KEY(TAB); MERIAN_KEY(BACKSPACE);
    MERIAN_KEY(INSERT); MERIAN_KEY(DELETE_KEY); MERIAN_KEY(RIGHT); MERIAN_KEY(LEFT);
    MERIAN_KEY(DOWN); MERIAN_KEY(UP); MERIAN_KEY(PAGE_UP); MERIAN_KEY(PAGE_DOWN); MERIAN_KEY(HOME);
    MERIAN_KEY(END); MERIAN_KEY(CAPS_LOCK); MERIAN_KEY(SCROLL_LOCK); MERIAN_KEY(NUM_LOCK);
    MERIAN_KEY(PRINT_SCREEN); MERIAN_KEY(PAUSE);
    MERIAN_KEY(F1); MERIAN_KEY(F2); MERIAN_KEY(F3); MERIAN_KEY(F4); MERIAN_KEY(F5); MERIAN_KEY(F6);
    MERIAN_KEY(F7); MERIAN_KEY(F8); MERIAN_KEY(F9); MERIAN_KEY(F10); MERIAN_KEY(F11);
    MERIAN_KEY(F12);
    MERIAN_KEY(KP_0); MERIAN_KEY(KP_1); MERIAN_KEY(KP_2); MERIAN_KEY(KP_3); MERIAN_KEY(KP_4);
    MERIAN_KEY(KP_5); MERIAN_KEY(KP_6); MERIAN_KEY(KP_7); MERIAN_KEY(KP_8); MERIAN_KEY(KP_9);
    MERIAN_KEY(KP_DECIMAL); MERIAN_KEY(KP_DIVIDE); MERIAN_KEY(KP_MULTIPLY);
    MERIAN_KEY(KP_SUBTRACT); MERIAN_KEY(KP_ADD); MERIAN_KEY(KP_ENTER); MERIAN_KEY(KP_EQUAL);
    MERIAN_KEY(LEFT_SHIFT); MERIAN_KEY(LEFT_CTRL); MERIAN_KEY(LEFT_ALT); MERIAN_KEY(LEFT_SUPER);
    MERIAN_KEY(RIGHT_SHIFT); MERIAN_KEY(RIGHT_CTRL); MERIAN_KEY(RIGHT_ALT);
    MERIAN_KEY(RIGHT_SUPER); MERIAN_KEY(MENU);
    // clang-format on
#undef MERIAN_KEY

    nb::enum_<InputController::KeyStatus>(m, "KeyStatus")
        .value("PRESS", InputController::KeyStatus::PRESS)
        .value("RELEASE", InputController::KeyStatus::RELEASE)
        .value("REPEAT", InputController::KeyStatus::REPEAT);

    nb::enum_<InputController::MouseButton>(m, "MouseButton")
        .value("MOUSE1", InputController::MouseButton::MOUSE1)
        .value("MOUSE2", InputController::MouseButton::MOUSE2)
        .value("MOUSE3", InputController::MouseButton::MOUSE3)
        .value("MOUSE4", InputController::MouseButton::MOUSE4)
        .value("MOUSE5", InputController::MouseButton::MOUSE5);

    nb::enum_<InputController::ModKey>(m, "ModKey", nb::is_flag(), nb::is_arithmetic())
        .value("SHIFT", InputController::SHIFT)
        .value("CONTROL", InputController::CONTROL)
        .value("ALT", InputController::ALT)
        .value("SUPER", InputController::SUPER)
        .value("CAPS_LOCK", InputController::CAPS_LOCK)
        .value("NUM_LOCK", InputController::NUM_LOCK);

    nb::enum_<WindowCursorShape>(m, "CursorShape")
        .value("ARROW", WindowCursorShape::Arrow)
        .value("TEXT_INPUT", WindowCursorShape::TextInput)
        .value("RESIZE_ALL", WindowCursorShape::ResizeAll)
        .value("RESIZE_NS", WindowCursorShape::ResizeNS)
        .value("RESIZE_EW", WindowCursorShape::ResizeEW)
        .value("RESIZE_NESW", WindowCursorShape::ResizeNESW)
        .value("RESIZE_NWSE", WindowCursorShape::ResizeNWSE)
        .value("HAND", WindowCursorShape::Hand)
        .value("NOT_ALLOWED", WindowCursorShape::NotAllowed)
        .value("HIDDEN", WindowCursorShape::Hidden);
}

} // namespace

void bind_window(nb::module_& m) {
    m.attr("HOST_WINDOWS_EXTENSION") = ExtensionHostWindows::name;

    bind_input_enums(m);

    nb::class_<HostWindow> window(
        m, "HostWindow",
        "The window of a Window node in a context with HOST_WINDOWS_EXTENSION: it presents into "
        "host memory and takes its extent and input from the host. Usable from any thread; the "
        "input reaches the graph in its next iteration.");

    nb::class_<HostCopy>(window, "Frame")
        .def_prop_ro("format", [](const HostCopy& copy) { return vk::to_string(copy.format); })
        .def_prop_ro("extent",
                     [](const HostCopy& copy) {
                         return nb::make_tuple(copy.extent.width, copy.extent.height);
                     })
        .def_prop_ro("array", &host_copy_array, nb::rv_policy::reference,
                     "The texels without a copy, (height, width, channels) in the order of format.")
        .def("__repr__", [](const HostCopy& copy) {
            return fmt::format("Frame({}x{}, {})", copy.extent.width, copy.extent.height,
                               vk::to_string(copy.format));
        });

    window.def_prop_ro("title", &HostWindow::get_title)
        .def_prop_ro("cursor", &HostWindow::get_cursor, "The cursor shape the graph asks for.")
        .def_prop_ro("mouse_grabbed", &HostWindow::is_mouse_grabbed,
                     "Whether the graph asks for raw mouse input: a hidden cursor whose "
                     "positions move without bounds.")
        .def_prop_ro("extent",
                     [](HostWindow& window) {
                         const vk::Extent2D extent = window.framebuffer_extent();
                         return nb::make_tuple(extent.width, extent.height);
                     })
        .def(
            "set_extent",
            [](HostWindow& window, const uint32_t width, const uint32_t height,
               const float display_scale) { window.set_extent({width, height}, display_scale); },
            "width"_a, "height"_a, "display_scale"_a = 1.f,
            "In pixels, display_scale pixels per logical pixel. 0x0 presents nothing.")
        .def("get_new_frame", &HostWindow::get_new_frame,
             "The newest frame that arrived since the last call, None if none did.")
        .def("send_cursor", &HostWindow::send_cursor, "x"_a, "y"_a, "In pixels.")
        .def("send_mouse_button", &HostWindow::send_mouse_button, "button"_a, "status"_a,
             "mods"_a = 0, "mods: ModKey flags.")
        .def("send_scroll", &HostWindow::send_scroll, "x"_a, "y"_a, "In steps.")
        .def("send_key", &HostWindow::send_key, "key"_a, "status"_a, "mods"_a = 0,
             "mods: ModKey flags.")
        .def("send_char", &HostWindow::send_char, "codepoint"_a, "Typed text, a Unicode codepoint.")
        .def("send_focus", &HostWindow::send_focus, "focused"_a)
        .def("request_close", &HostWindow::request_close,
             "As the close button of a window does; the Window node reacts per its properties.");
}

} // namespace merian::python
