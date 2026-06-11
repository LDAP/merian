## Utilities

Merian provides utilities for common Vulkan programming tasks. See `include/merian/utils/` and `include/merian/vk/utils/` for the full list.

### Camera (`utils/camera/`)

- `Camera`: Computes view and projection matrices from position, target and up vector.
- `CameraAnimator`: Smoothly interpolates camera motion over time.
- `FlyCameraController`: Backend-agnostic free-fly camera control driven by an `InputController` (WASD + right-mouse look).

### Configuration (`utils/properties.hpp`)

`Properties` is an "immediate-mode" configuration API with implementations for:
- ImGUI display (`properties_imgui.hpp`)
- JSON dump (`properties_json_dump.hpp`)
- JSON load (`properties_json_load.hpp`)

Nodes expose configuration by overriding `properties(Properties& props)`. The graph calls this and routes it to ImGUI or JSON.

### FileLoader

Searches for files across a list of registered search paths. Merian registers the project's source and install paths automatically. The context exposes its file loader via `context->get_file_loader()`.

### InputController (`utils/input_controller.hpp`)

Abstract keyboard and mouse input interface. `input_controller_glfw.hpp` provides a GLFW-backed implementation.

### Profiler (`vk/utils/profiler.hpp`)

CPU and GPU profiling. Enabled at compile time with:

```bash
meson configure build -Dmerian:performance_profiling=true
```

Accessible via `GraphRun::get_profiler()` (may be `nullptr` if disabled).

### ThreadPool (`utils/concurrent/`)

A simple work-stealing thread pool. Merian initializes a default thread pool accessible via `GraphRun::get_thread_pool()`.

### CPUQueue (`vk/utils/cpu_queue.hpp`)

A thread-safe task queue for off-thread CPU work (e.g. file I/O, mesh loading). Accessible via `GraphRun::get_cpu_queue()`.

### RenderDoc (`utils/renderdoc.hpp`)

Helper class to trigger RenderDoc frame captures from code:

```cpp
merian::Renderdoc renderdoc;
renderdoc.start_frame_capture();
// ... frame ...
renderdoc.end_frame_capture();
```

### Other utilities

- `Stopwatch` — wall-clock timing.
- `RingBuffer<T>` — fixed-size circular buffer.
- `XorShift` — fast PRNG.
- `normal_encoding.hpp` — CPU-side octahedral normal encode/decode.
- `math.hpp`, `vector_matrix.hpp` — GLM-compatible math helpers.
- `string.hpp` — string formatting utilities.
