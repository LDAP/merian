## Windows

A `Window` comes from a `WindowProvider` context extension and creates the swapchain that presents
to it. Its input reaches the `InputController` of `get_input_controller()` in `poll_events()`.

| Extension | Provider | Presents to | Priority |
|---|---|---|---|
| `"merian-host-windows"` | `ExtensionHostWindows` | host memory of a `HostWindow` | 100, not auto-loaded |
| `"SDL3 (Video Subsystem)"` | `ExtensionSDLVideo` | an SDL window | 50 |
| `"GLFW"` | `ExtensionGLFW` | a GLFW window | 40 |

```c++
const WindowHandle window =
    context->find_provider<WindowProvider>()->create_window(context->get_device());
SwapchainManager swapchain_manager(window->create_swapchain(context));
```

The `Window` node presents through the provider its `backend` property selects; `auto`, the default,
is the provider of the highest priority.

### Host windows

A `HostWindow` is shown by its host application, e.g. the views of the Python graph editor. Its
`HostSwapchain` presents in the transfer-source layout (`Swapchain::get_present_layout()`) by copying
the image into host memory, every iteration as with Mailbox. The host takes the newest frame with
`get_new_frame()` at its own pace; the frames it skipped are dropped. The host sets the extent with
`set_extent()` (0x0 presents nothing) and sends input with `send_*()`, which the window dispatches
in its next `poll_events()`. `is_mouse_grabbed()` and `get_cursor()` tell the host what the graph
asks for. The window's clipboard is its own and fullscreen has no effect.

The `Window` node names its window "merian: <identifier>". In Python, `Graph.window(identifier)`
returns the `HostWindow` of a `Window` node in a context that loaded
`merian.graph.HOST_WINDOWS_EXTENSION`, `Graph.windows` those of all of them.
